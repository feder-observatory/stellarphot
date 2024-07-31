from enum import Enum


class AAVSOFilters(str, Enum):
    """
    The definitive list of AAVSO filters is at https://www.aavso.org/filters
    """

    U = "U"
    B = "B"
    V = "V"
    RJ = "RJ"
    Rc = "R"
    Ic = "I"
    IJ = "IJ"
    J = "J"
    H = "H"
    K = "K"
    TG = "TG"
    TB = "TB"
    TR = "TR"
    CV = "CV"
    CR = "CR"
    SZ = "SZ"
    SU = "SU"
    SG = "SG"
    SR = "SR"
    SI = "SI"
    STU = "STU"
    STV = "STV"
    STB = "STB"
    STY = "STY"
    STHBW = "STHBW"
    STHBN = "STHBN"
    MA = "MA"
    MB = "MB"
    MI = "MI"
    ZS = "ZS"
    Y = "Y"
    HA = "HA"
    HAC = "HAC"
    CBB = "CBB"
    O = "O"  # noqa: E741
