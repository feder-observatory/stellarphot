from typing import Annotated

import pytest
from astropy.units import Quantity, Unit, get_physical_type
from pydantic import BaseModel, ValidationError

from stellarphot.settings.astropy_pydantic import (
    EquivalentTo,
    QuantityType,
    UnitType,
    WithPhysicalType,
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
        "meter",
        "parsec / fortnight",
        "",
    ],
)
def test_unit_initialization(init):
    # Make sure we can initialize from each of the ways a Unit can
    # be initiailized
    expected = Unit(init)
    unit = _UnitModel(unit=init)
    assert expected == unit.unit


@pytest.mark.parametrize(
    "init",
    [
        -2 * Unit("m"),
        1,
        "5 meter",
        "13 parsec / fortnight",
        "42",
    ],
)
def test_quantity_intialization(init):
    # Make sure we can initialize from each of the ways a Quantity can
    # be initiailized
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


def test_quantity_type_with_invalid_quantity():
    with pytest.raises(ValidationError, match="It does not start with a number"):
        _QuantityModel(quantity="meter")
