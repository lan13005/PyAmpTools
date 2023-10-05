from test_files import parMgr
import pytest

@pytest.mark.parmgr
def test_parMgr():
    nProdPars, prefit_nll, par_real, par_imag, post_nll = parMgr.runTest()
    assert( nProdPars == 6 )
    assert( prefit_nll != 1e6 and prefit_nll is not None )
    assert( par_real == 15 and par_imag == 0 )
    assert( post_nll != 1e6 and post_nll is not None )
