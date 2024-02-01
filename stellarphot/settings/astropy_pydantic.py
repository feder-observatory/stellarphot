from dataclasses import dataclass
from typing import Annotated, Any

from astropy.units import (
    PhysicalType,
    Quantity,
    Unit,
    UnitBase,
    UnitConversionError,
    get_physical_type,
)
from pydantic import (
    GetCoreSchemaHandler,
)
from pydantic_core import core_schema

__all__ = [
    "UnitType",
    "QuantityType",
    "EquivalentTo",
    "WithPhysicalType",
    "AstropyValidator",
]

_PHYSICAL_TYPES_URL = "https://docs.astropy.org/en/stable/units/ref_api.html#module-astropy.units.physical"

# Approach to validation of units was inspired by the GammaPy project
# which did it before we did:
# https://docs.gammapy.org/dev/_modules/gammapy/analysis/config.html

# Update for pydantic 2.0, based on the pydantic docs:
#   https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types


# This function will end up creating the instance from the input value.
def validate_by_instantiating(source_type):
    """
    Return a function that tries to create an instance of source_type from a value.
    The intended use of this is as a vallidotr in pydantic.

    Parameters
    ----------

    source_type : Any
        The type to create an instance of.

    Returns
    -------
    function
        A function that tries to create an instance of source_type from a value.
    """

    def _validator(value):
        # If the value is valid we will be able to create an instance of the
        # source_type from it. For example, if source_type is astropy.units.Unit,
        # then we should be able to create a Unit from the value.
        try:
            result = source_type(value)
        except TypeError as err:
            # Need to raise a ValueError for pydantic to catch it as a
            # validation error.
            raise ValueError(str(err)) from err
        return result

    return _validator


class _UnitQuantTypePydanticAnnotation:
    """
    This class is used to annotate fields where validation consists of checking
    whether an instance can be created.

    In astropy, this includes `astropy.units.Unit` and `astropy.units.Quantity`.

    We return a pydantic_core.CoreSchema that behaves in the following ways:
        * When initializing from python:
            * A Unit or a Quantity will pass validation and be returned as-is
            * A string or a float will be used to create a Unit or a Quantity.
            * Nothing else will pass validation
        * When initializing from json:
            * The Unit or Quantity must be represented as a string in the json.
        * Serialization will always represent the Unit or Quantity as a string.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """
        We return a pydantic_core.CoreSchema that behaves in the following ways:

        * A Unit or a Quantity will pass validation and be returned as-is
        * A string or a float will be used to create a Unit or a Quantity.
        * Nothing else will pass validation
        * Serialization will always return just a string
        """
        # Both Unit and Quantity can be created from a string or a float or an
        # instance of the same type. So we need to check for all of those cases.

        # core_schema.chain_schema runs the value through each of the schema
        # in the list, in order. The output of one schema is the input to the next.

        # The schema below validates a Unit or Quantity from a string. The first link
        # in the chain is a schema that validates a string. The second link is a
        # schema that takes that string value and creates a Unit or Quantity from it.

        # This schema is used in three places by pydantic:
        #   1. When validating a python value
        #   2. When validating a json value
        #   3. When constructing a JSON schema for the model
        from_str_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(
                    validate_by_instantiating(source_type)
                ),
            ]
        )

        # This schema validates a Unit or Quantity from a float. The first link in the
        # chain is a schema that validates the input as a number. The second link is a
        # schema that takes that numeric value and creates a Unit or Quantity from it.

        # This schema is used in just one place by pydantic:
        #   1. When validating a python value
        from_float_schema = core_schema.chain_schema(
            [
                core_schema.float_schema(),
                core_schema.no_info_plain_validator_function(
                    validate_by_instantiating(source_type)
                ),
            ]
        )

        return core_schema.json_or_python_schema(
            # The next line specifies two things:
            #   1. The schema used to validate value from JSON. Since we are using the
            #      schema for a string value, the values in the JSON file must be
            #      strings, even though something like "{quantity: 1}" i.e. initializing
            #      from a number, would work in python. The reason for this choice is
            #      that the serialization is to a string, so that is what we expect from
            #      JSON. If we wanted to allow initialization from a number in JSON, we
            #      would need to use a union schema that consisted of
            #      from_str_schema and from_float_schema.
            #   2. The schema used to construct a JSON schema for the model. When you do
            #      `model_json_schema` with a `chain_schema`, then the first entry of
            #      the chain is used if `mode="validation"` and the last entry of the
            #      chain is used if `mode="serialization"`. I guess this makes sense,
            #      since the first thing in the chain has to handle the value coming
            #      from json,while the last thing generates the python value for the
            #      input. With the choice below we will *always* want`mode="validation"`
            #      because pydantic cannot generate a schema fora Unit or Quantity.
            json_schema=from_str_schema,
            # union_schema takes a list of schemas and returns a schema that
            # is the "best" match. See the link below for a description of
            # what counts as "best":
            #   https://docs.pydantic.dev/dev/concepts/unions/#smart-mode
            #
            # In short, schemas are tried from left-to-right, and an exact type match
            # wins.
            #
            # The construction below tries to make a value starting from a Unit,
            # a Quantity, a string, or a float. The first two are instances, so we
            # use is_instance_schema.
            python_schema=core_schema.union_schema(
                [
                    # Check if it's an instance first before doing any further work.
                    # Would be nice to provide a list of classes here instead
                    # of one-by-one.
                    core_schema.is_instance_schema(UnitBase),
                    core_schema.is_instance_schema(Quantity),
                    from_str_schema,
                    from_float_schema,
                ]
            ),
            # Serialize by converting the Unit or Quantity to a string.
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: str(instance)
            ),
        )


@dataclass
class EquivalentTo:
    """
    This class is a pydantic "marker" (their word for this kind of thing) that
    can be used to annotate fields that should be equivalent to a given unit.

    Parameters
    ----------
    equivalent_unit : `astropy.units.Unit`
        The unit that the annotated field should be equivalent to.

    Examples
    --------
    >>> from typing import Annotated
    >>> from pydantic import BaseModel, ValidationError
    >>> from stellarphot.settings.astropy_pydantic import (
    ...     WithPhysicalType,
    ...     UnitType,
    ...     QuantityType
    ... )
    >>> class UnitModel(BaseModel):
    ...     length: Annotated[UnitType, EquivalentTo("m")]
    >>> UnitModel(length="lightyear")
    UnitModel(length=Unit("lyr"))
    >>> try:
    ...     UnitModel(length="second")
    ... except ValidationError as err:
    ...     print(err)
    1 validation error for UnitModel
    length
      Value error, Unit s is not equivalent to m...
    >>> # Next let's do a Quantity
    >>> class QuantityModel(BaseModel):
    ...     velocity: Annotated[QuantityType, EquivalentTo("m / s")]
    >>> QuantityModel(velocity="3 lightyear / year")
    QuantityModel(velocity=<Quantity 3. lyr / yr>)
    >>> try:
    ...     QuantityModel(velocity="3 parsec / lightyear")
    ... except ValidationError as err:
    ...     print(err)
    1 validation error for QuantityModel
    velocity
      Value error, Unit pc / lyr is not equivalent to m / s...
    """

    equivalent_unit: Unit
    """Unit that the annotated field should be equivalent to."""

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ):
        def check_equivalent(value):
            # We are getting either a Unit or a Quantity. If it's a Quantity, we
            # need to get the unit from it.
            if isinstance(value, UnitBase):
                value_unit = value
            else:
                value_unit = value.unit

            try:
                value.to(self.equivalent_unit)
            except UnitConversionError:
                # Raise a ValueError for pydantic to catch it as a validation error.
                raise ValueError(
                    f"Unit {value_unit} is not equivalent to {self.equivalent_unit}"
                ) from None
            return value

        # Calling handler(source_type) will pass the result of this annotation
        # to the next annotation in the chain.
        return core_schema.no_info_after_validator_function(
            check_equivalent, handler(source_type)
        )

    def __hash__(self):
        return hash(self.equivalent_unit)


@dataclass
class WithPhysicalType:
    """
    This class is a pydantic "marker" (their word for this kind of thing) that
    can be used to annotate fields that should be of a specific physical type.

    Parameters
    ----------
    physical_type : str or `astropy.units.physical.PhysicalType`
        The physical type of the annotated field.

    Examples
    --------
    >>> from typing import Annotated
    >>> from pydantic import BaseModel, ValidationError
    >>> from stellarphot.settings.astropy_pydantic import (
    ...     WithPhysicalType,
    ...     UnitType,
    ...     QuantityType
    ... )
    >>> class UnitModel(BaseModel):
    ...     length: Annotated[UnitType, WithPhysicalType("length")]
    >>> UnitModel(length="meter")
    UnitModel(length=Unit("m"))
    >>> try:
    ...     UnitModel(length="second")
    ... except ValidationError as err:
    ...     print(err)
    1 validation error for UnitModel
    length
      Value error, Unit of s is not equivalent to length...
    >>> # Next let's do a Quantity
    >>> class QuantityModel(BaseModel):
    ...     velocity: Annotated[QuantityType, WithPhysicalType("speed")]
    >>> QuantityModel(velocity="3 meter / second")
    QuantityModel(velocity=<Quantity 3. m / s>)
    >>> try:
    ...     QuantityModel(velocity="3 meter")
    ... except ValidationError as err:
    ...     print(err)
    1 validation error for QuantityModel
    velocity
      Value error, Unit of 3.0 m is not equivalent to speed...
    """

    physical_type: str | PhysicalType

    def __post_init__(self):
        try:
            get_physical_type(self.physical_type)
        except ValueError as err:
            # Add a link to the astropy documentation for physical types
            # to the error message.
            raise ValueError(
                str(err)
                + f"\nSee {_PHYSICAL_TYPES_URL} for a list of valid physical types."
            ) from err

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ):
        def check_physical_type(value):
            is_same = get_physical_type(value) == get_physical_type(self.physical_type)
            if is_same:
                return value
            else:
                raise ValueError(
                    f"Unit of {value} is not equivalent to {self.physical_type}"
                ) from None

        # As in the EquivalentTo annotation, calling handler(source_type) will pass
        # the result of this annotation to the next annotation in the chain.
        return core_schema.no_info_after_validator_function(
            check_physical_type, handler(source_type)
        )

    def __hash__(self):
        return hash(self.physical_type)


# We have lost default titles and exmples, but that is maybe not so bad

# This is really nice compared to pydantiv v1...
UnitType = Annotated[Unit, _UnitQuantTypePydanticAnnotation]

# Quantity type is really clean too
QuantityType = Annotated[Quantity, _UnitQuantTypePydanticAnnotation]


# It turns out all (almost all?) astropy types have, buried in them, a representation
# that is a python dictionary info._represent_as_dict and info._construct_from_dict.
# This is what is used to represent astropy
# objects in Tables and FITS files. So we can use this to create a json schema
# for the astropy types.
def serialize_astropy_type(value):
    """
    Two things might happen here:

    1. value serializes to JSON because each value in the dict reperesentation
        is a type JSON knows how to represent, or
    2. value does not serialize because one or more of the values in the dict
        representation is itself an astropy class.
    """

    def dict_rep(instance):
        return instance.info._represent_as_dict()

    if isinstance(value, UnitBase | Quantity):
        return str(value)
    try:
        rep = dict_rep(value)
    except AttributeError:
        # Either this is not an astropy thing, in which case just return the
        # value, or this is an astropy unit. Happily, we can already serialize
        # that.
        return value if not hasattr(value, "to_string") else value.to_string()

    result = {}
    for k, v in rep.items():
        result[k] = serialize_astropy_type(v)

    return result


class AstropyValidator:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type,
        _handler,
    ):
        def astropy_object_from_dict(value):
            """
            This is NOT the right way to be doing this when there are nested
            definitions, e.g. in a SkyCoord where the RA and Dec are each
            an angle, which is not a native python type.
            """
            return source_type.info._construct_from_dict(value)

        from_dict_schema = core_schema.chain_schema(
            [
                core_schema.dict_schema(),
                core_schema.no_info_plain_validator_function(astropy_object_from_dict),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_dict_schema,
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(source_type),
                    from_dict_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_astropy_type
            ),
        )
