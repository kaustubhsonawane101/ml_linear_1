from ml_linear import train_model
def test_ml():
    r2=train_model()
    assert r2 > 0.5