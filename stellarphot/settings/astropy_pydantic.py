from astropy.units import Quantity, Unit

__all__ = ["UnitType", "QuantityType", "PixelScaleType"]

# Approach to validation of units was inspired by the GammaPy project
# which did it before we did:
# https://docs.gammapy.org/dev/_modules/gammapy/analysis/config.html


class UnitType(Unit):
    # Validator for Unit type
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return Unit(v)

    @classmethod
    def __modify_schema__(cls, field_schema, field):
        # Set default values for the schema in case the field doesn't provide them
        name = "Unit"
        description = "An astropy unit"

        name = field.name or name
        description = field.field_info.description or description
        examples = field.field_info.extra.get("examples", [])

        field_schema.update(
            {
                "title": name,
                "description": description,
                "examples": examples,
                "type": "string",
            }
        )


class QuantityType(Quantity):
    # Validator for Quantity type
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        try:
            v = Quantity(v)
        except TypeError as err:
            raise ValueError(f"Invalid value for Quantity: {v}") from err
        else:
            if not v.unit.bases:
                raise ValueError("Must provided a unit")
        return v

    @classmethod
    def __modify_schema__(cls, field_schema, field):
        # Set default values for the schema in case the field doesn't provide them
        name = "Quantity"
        description = "An astropy Quantity with units"

        name = field.name or name
        description = field.field_info.description or description
        examples = field.field_info.extra.get("examples", [])

        field_schema.update(
            {
                "title": name,
                "description": description,
                "examples": examples,
                "type": "string",
            }
        )


class PixelScaleType(Quantity):
    # Validator for pixel scale type
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        try:
            v = Quantity(v)
        except TypeError as err:
            raise ValueError(f"Invalid value for Quantity: {v}") from err
        if (
            len(v.unit.bases) != 2
            or v.unit.bases[0].physical_type != "angle"
            or v.unit.bases[1].name != "pix"
        ):
            raise ValueError(f"Invalid unit for pixel scale: {v.unit!r}")
        return v

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            {
                "title": "PixelScale",
                "description": "An astropy Quantity with units of angle per pixel",
                "examples": ["0.563 arcsec / pix"],
                "type": "string",
            }
        )
