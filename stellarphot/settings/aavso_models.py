from enum import StrEnum


class AAVSOFilters(StrEnum):
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
    GG = "GG"  # This is in VSX -- it is Gaia G
    GBP = "GBP"  # This is not in VSX -- it is Gaia BP
    GRP = "GRP"  # This is not in VSX -- it is Gaia RP
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
