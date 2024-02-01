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

__all__ = ["UnitType", "QuantityType", "EquivalentTo", "WithPhysicalType"]

_PHYSICAL_TYPES_URL = "https://docs.astropy.org/en/stable/units/ref_api.html#module-astropy.units.physical"

# Approach to validation of units was inspired by the GammaPy project
# which did it before we did:
# https://docs.gammapy.org/dev/_modules/gammapy/analysis/config.html

# Update for pydantic 2.0, based on the pydantic docs:
#   https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types


class _UnitQuantTypePydanticAnnotation:
    """
    This class is used to annotate fields where validation consists of checking
    whether an instance can be created.

    In astropy, this includes `astropy.units.Unit` and `astropy.units.Quantity`.
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

        def validate_by_instantiating(value):
            # If the value is valid we will be able to create an instance of the
            # source_type from it. For example, if source_type is astropy.units.Unit,
            # then we should be able to create a Unit from the value.
            try:
                result = source_type(value)
            except TypeError as err:
                raise ValueError(str(err)) from err
            return result

        # Both Unit and Qunatity can be created from a string or a float or an
        # instance of the same type. So we need to check for all of those cases.

        # core_schema.chain_schema runs the value through each of the schema
        # in the list, in order. The output of one schema is the input to the next.

        # When you do `model_json_schema` with a `chain_schema`, then the first entry is
        # used if `mode="validation"` and the last is used if `mode="serialization"`
        # from the schema used to serialize json.

        # I guess this makes sense, since the first thing in the chain has to handle the
        # value coming from json, while the last thing generates the python value for
        # the input.
        from_str_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(validate_by_instantiating),
            ]
        )

        from_float_schema = core_schema.chain_schema(
            [
                core_schema.float_schema(),
                core_schema.no_info_plain_validator_function(validate_by_instantiating),
            ]
        )
        return core_schema.json_or_python_schema(
            json_schema=from_str_schema,
            # union_schema takes a list of schemas and returns a schema that
            # is the "best" match. See the link below for a description of
            # what counts as "best":
            #   https://docs.pydantic.dev/dev/concepts/unions/#smart-mode
            #
            # In short, schemas are tried from left-to-right, and an exact type match
            # wins.
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
            # Serialization by converting to a string.
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: str(instance)
            ),
        )


@dataclass
class EquivalentTo:
    equivalent_unit: Unit

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ):
        def check_equivalent(value):
            if isinstance(value, UnitBase):
                value_unit = value
            else:
                value_unit = value.unit

            try:
                value.to(self.equivalent_unit)
            except UnitConversionError:
                raise ValueError(
                    f"Unit {value_unit} is not equivalent to {self.equivalent_unit}"
                ) from None
            return value

        return core_schema.no_info_after_validator_function(
            check_equivalent, handler(source_type)
        )


@dataclass
class WithPhysicalType:
    physical_type: str | PhysicalType

    def __post_init__(self):
        try:
            get_physical_type(self.physical_type)
        except ValueError as err:
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

        return core_schema.no_info_after_validator_function(
            check_physical_type, handler(source_type)
        )


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
