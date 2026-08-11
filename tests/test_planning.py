from sparseecho import compile_query_plan


def test_plan_fingerprint_is_stable_and_order_sensitive():
    a = compile_query_plan(n_views=16, ordering="gray")
    b = compile_query_plan(n_views=16, ordering="gray")
    c = compile_query_plan(n_views=16, ordering="binary")
    assert a.fingerprint() == b.fingerprint()
    assert len(a.fingerprint()) == 64
    assert a.fingerprint() != c.fingerprint()
    assert a.to_dict()["plan_fingerprint"] == a.fingerprint()
