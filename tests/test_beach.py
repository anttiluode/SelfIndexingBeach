import numpy as np
from beach import Beach, sensitivity_matrix
from experiment import run, PROBES


def test_weak_reads_do_not_write():
    b=Beach();b.write('A')
    original=b.sand.copy()
    x=b.fingerprint(PROBES)
    assert np.isfinite(x).all()
    assert np.array_equal(b.sand,original)


def test_frozen_sand_control():
    b=Beach(plastic=False)
    y=b.fingerprint(PROBES)
    b.write('A')
    assert np.array_equal(b.sand,np.zeros(b.p.n))
    assert np.array_equal(y,b.fingerprint(PROBES))


def test_write_measurably_changes_bed_and_echo():
    b=Beach();y0=b.fingerprint(PROBES)
    b.write('A')
    assert np.max(b.sand) > .01
    assert np.linalg.norm(b.fingerprint(PROBES)-y0)>1e-4


def test_order_effect_is_real_but_hard_to_read():
    ab=Beach();ba=Beach()
    for x in 'AB':ab.write(x)
    for x in 'BA':ba.write(x)
    assert 0 < np.linalg.norm(ab.sand-ba.sand)<0.001
    assert 0 < np.linalg.norm(ab.fingerprint(PROBES)-ba.fingerprint(PROBES))<1e-5


def test_extra_probes_add_material_observability():
    b=Beach();b.write('A');b.write('B')
    j1=sensitivity_matrix(b,(PROBES[1],))
    j4=sensitivity_matrix(b,PROBES)
    s1=np.linalg.svd(j1,compute_uv=False)
    s4=np.linalg.svd(j4,compute_uv=False)
    assert (s4>1e-4).sum()>(s1>1e-4).sum()


def test_current_echo_helps_predict_next_echo():
    r=run(train_episodes=25,test_episodes=10,steps=5,seed=13,plot=False)
    assert r['read']['mse'] < .2*r['event']['mse']
    assert r['frozen_max_sand_change']==0
    assert r['frozen_max_echo_change']==0
