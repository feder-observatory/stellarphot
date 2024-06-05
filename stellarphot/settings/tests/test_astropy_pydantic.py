import json
from typing import Annotated

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.units import Quantity, Unit, get_physical_type
from pydantic import BaseModel, ValidationError

from stellarphot.settings.astropy_pydantic import (
    AstropyValidator,
    EquivalentTo,
    QuantityType,
    UnitType,
    WithPhysicalType,
    serialize_astropy_type,
)


class _UnitModel(BaseModel):
    # Dummy class for testing
    unit: UnitType


class _QuantityModel(BaseModel):
    # Dummy class for testing
    quantity: QuantityType


@pytest.mark.parametrize(
    "init",
    [
        Unit("m"),
        1,
        4.0,
        "meter",
        "parsec / fortnight",
        "",
    ],
)
def test_unit_initialization(init):
    # Make sure we can initialize from each of the ways a Unit can
    # be initialized
    expected = Unit(init)
    unit = _UnitModel(unit=init)
    assert expected == unit.unit


@pytest.mark.parametrize(
    "init",
    [
        -2 * Unit("m"),
        -2.0 * Unit("m"),
        1,
        1.0,
        "5 meter",
        "13 parsec / fortnight",
        "42",
    ],
)
def test_quantity_intialization(init):
    # Make sure we can initialize from each of the ways a Quantity can
    # be initialized
    expected = Quantity(init)
    quantity = _QuantityModel(quantity=init)
    assert expected == quantity.quantity


class _ModelEquivalentTo(BaseModel):
    unit_meter: Annotated[UnitType, EquivalentTo(equivalent_unit="m")]
    quantity_meter: Annotated[QuantityType, EquivalentTo(equivalent_unit="m")]


class _ModelWithPhysicalType(BaseModel):
    quant_physical_length: Annotated[QuantityType, WithPhysicalType("length")]
    unit_physical_time: Annotated[UnitType, WithPhysicalType("time")]


def test_equivalent_to():
    # Make sure we can annotate with an equivalent unit

    # This should succeed
    model = _ModelEquivalentTo(
        unit_meter="km",
        quantity_meter=Quantity(1, "mm"),
    )
    assert model.unit_meter == Unit("km")
    assert model.quantity_meter.to("m").value == model.quantity_meter.value * 1e-3

    # Now some failures

    with pytest.raises(ValidationError, match="Unit s is not equivalent to"):
        _ModelEquivalentTo(unit_meter="km", quantity_meter=Quantity("1 s"))

    with pytest.raises(ValidationError, match="Unit s is not equivalent to"):
        _ModelEquivalentTo(unit_meter="s", quantity_meter=Quantity("1 m"))


def test_equivalent_to_can_be_used_in_union():
    class ModelWithUnion(BaseModel):
        may_be_none: Annotated[QuantityType, EquivalentTo("second")] | None

    model = ModelWithUnion(may_be_none=None)
    assert model.may_be_none is None
    model = ModelWithUnion(may_be_none=Quantity(1, "s"))
    assert model.may_be_none == Quantity(1, "s")


def test_physical_type_can_be_used_in_union():
    class ModelWithUnion(BaseModel):
        may_be_none: Annotated[QuantityType, WithPhysicalType("time")] | None

    model = ModelWithUnion(may_be_none=None)
    assert model.may_be_none is None
    model = ModelWithUnion(may_be_none=Quantity(1, "s"))
    assert model.may_be_none == Quantity(1, "s")


def test_with_physical_type():
    # Make sure we can annotate with a physical type
    model = _ModelWithPhysicalType(
        quant_physical_length=Quantity(1, "m"),
        unit_physical_time=17 * Unit("second"),
    )

    assert get_physical_type(model.quant_physical_length) == "length"
    assert get_physical_type(model.unit_physical_time) == "time"

    # Now some failures
    # Pass a time in for physical type of length
    with pytest.raises(
        ValidationError, match="Unit of 1.0 s is not equivalent to length"
    ):
        _ModelWithPhysicalType(
            quant_physical_length=Quantity(1, "s"),
            unit_physical_time=Unit("second"),
        )

    # Pass a length in for physical type of time
    with pytest.raises(ValidationError, match="Unit of m is not equivalent to time"):
        _ModelWithPhysicalType(
            quant_physical_length=Quantity(1, "m"),
            unit_physical_time=Unit("meter"),
        )


def test_bad_physical_type_raises_error():
    with pytest.raises(ValueError, match="'not_a_type' is not a known physical type"):

        class Model(BaseModel):
            not_a_type: Annotated[QuantityType, WithPhysicalType("not_a_type")]


def test_quantity_type_with_invalid_quantity():
    with pytest.raises(ValidationError, match="It does not start with a number"):
        _QuantityModel(quantity="meter")


@pytest.mark.parametrize(
    "input_json_string",
    [
        '{"quantity": "1 m"}',
        '{"quantity": "1"}',
        '{"quantity": "3 second"}',
    ],
)
def test_initialize_quantity_with_json(input_json_string):
    # Make sure we can initialize a Quantity from a json string
    # where the quantity value is stored in the json as a string.
    model = _QuantityModel.model_validate_json(input_json_string)
    model_json = json.loads(model.model_dump_json())
    input_json = json.loads(input_json_string)

    assert Quantity(model_json["quantity"]) == Quantity(input_json["quantity"])


def test_initialize_quantity_with_json_invalid():
    # Make sure we get an error when the json string is has a value
    # that is a float (same fail happens for integer).
    # Since our json validation assumes the value is a string, this
    # should fail.
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        _QuantityModel.model_validate_json('{"quantity": 14.0}')


@pytest.mark.parametrize(
    "input_json_string",
    [
        '{"unit": "m"}',
        '{"unit": "1"}',
        '{"unit": "parsec / fortnight"}',
    ],
)
def test_initialize_unit_with_json(input_json_string):
    # Make sure we can initialize a Unit from a json string
    # where the quantity value is stored in the json as a string.
    model = _UnitModel.model_validate_json(input_json_string)
    model_json = json.loads(model.model_dump_json())
    input_json = json.loads(input_json_string)

    assert Unit(model_json["unit"]) == Unit(input_json["unit"])


def test_initialize_unit_with_json_invalid():
    # Make sure we get an error when the json string is has a value
    # that is a float (same fail happens for integer).
    # Since our json validation assumes the value is a string, this
    # should fail.
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        _UnitModel.model_validate_json('{"unit": 14.0}')


@pytest.mark.parametrize(
    "klass,input",
    [
        (Time, "2021-01-01T00:00:00"),
        (SkyCoord, "00h42m44.3s +41d16m9s"),
    ],
)
def test_time_quant_pydantic(klass, input):
    class Model(BaseModel):
        value: Annotated[klass, AstropyValidator]

    val = klass(input)
    model = Model(value=val)

    # Value should be correct
    assert model.value == val

    # model dump should fully serialize to standard python types
    assert model.model_dump()["value"] == serialize_astropy_type(val)

    # We should be able to create a new model from the dumped json...
    model2 = Model.model_validate_json(model.model_dump_json())

    if klass is SkyCoord:
        np.testing.assert_almost_equal(
            model2.value.separation(model.value).arcsec,
            0,
            decimal=10,
        )
    else:
        assert model2.value == model.value


def test_time_pydantic_invalid_value():
    class Model(BaseModel):
        value: Annotated[Time, AstropyValidator]

    with pytest.raises(ValidationError, match="Input should be an instance of Time"):
        Model(value="not a time")
