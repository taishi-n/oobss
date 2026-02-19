from oobss import (
    AuxIVA,
    ILRMA,
    OnlineAuxIVA,
    OnlineILRMA,
    OnlineISNMF,
)


def test_public_imports() -> None:
    assert AuxIVA is not None
    assert ILRMA is not None
    assert OnlineAuxIVA is not None
    assert OnlineILRMA is not None
    assert OnlineISNMF is not None
