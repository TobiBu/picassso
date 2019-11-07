import picasso
import numpy as np
from nose.tools import assert_equals

def setup():
    global survey
    survey = picasso.load("test_data/")


def teardown():
    global survey
    del survey


def test_construct():
    """Check the basic properties of the survey"""
    assert (np.size(survey._files) ==1)
    assert(survey._num_galaxies == 21)
    assert (survey.path == "test_data/")
    for f in survey._files:
        assert(f.mode == 'r')
        

def test_properties():
    assert "image_res" in survey.properties


def test_loadable():
    """Check we have found all the blocks that should be in the survey"""
    blocks = survey.loadable_keys()
    expected_all = ['stars_GFM_StellarFormationTime','galaxy','stars_Masses','stars_GFM_Metallicity']

    assert(set(survey.loadable_keys()) == set(expected_all))


def test_standard_arrays():
    """Check we can actually load some of these arrays"""
    survey['galaxy']
    survey['galaxy'][0].properties['stars_Masses']

    galaxy_blocks = survey['galaxy'][0].properties.keys()
    for key in galaxy_blocks:
        survey['galaxy'][0].properties[key]

def test_array_sizes():
    """Check we have the right sizes for the arrays"""
    assert(np.shape(survey['galaxy'][0].properties['stars_Masses']) == (256, 256))
    assert(np.shape(survey['galaxy']) == (21,))
    assert(survey['galaxy'].dtype == dtype('O'))
    assert(survey['galaxy'][0].properties['stars_Masses'].dtype == dtype('float64'))
    assert(survey['galaxy'][0].properties['dm_mass'].dtype == dtype('float64'))
    assert(survey['galaxy'][0].properties['Galaxy_id'] == '12c1')






###### further tests from pynbody... check later which ones are needed...
def test_array_contents():
    """Check some array elements"""
    assert(np.max(survey["iord"]) == 8192)
    assert(np.min(survey["iord"]) == 1)
    assert(np.mean(survey["iord"]) == 4096.5)

    # 10/11/13 - AP - suspect the following tests are incorrect
    # because ordering of file did not agree with pynbody ordering

    assert(abs(np.mean(survey["pos"]) - 1434.664) < 0.004)
    assert(abs(survey["pos"][52][1] - 456.69678) < 0.001)
    assert(abs(survey.gas["u"][100] - 438.39496) < 0.001)
    assert(abs(survey.dm["mass"][5] - 0.04061608) < 0.001)


def test_fam_sim():
    """Check that an array loaded as families is the same as one loaded as a simulation array"""
    survey2 = pynbody.load("testdata/test_g2_survey")
    survey3 = pynbody.load("testdata/test_g2_survey")
    survey3.gas["pos"]
    survey3.dm["pos"]
    survey3.star["pos"]
    assert((survey3["pos"] == survey2["pos"]).all())


def test_write():
    """Check that we can write a new survey and read it again,
    and the written and the read are the same."""
    survey.write(filename='testdata/test_gadget_write')
    survey3 = pynbody.load('testdata/test_gadget_write')
    assert(set(survey.loadable_keys()) == set(survey3.loadable_keys()))
    assert((survey3["pos"].view(np.ndarray) == survey["pos"]).all())
    assert((survey3.gas["rho"].view(np.ndarray) == survey.gas["rho"]).all())
    assert(survey3.check_headers(survey.header, survey3.header))


def test_write_single_array():
    """Check that we can write a single array and read it back"""
    survey["pos"].write(overwrite=True)
    survey6 = pynbody.load("testdata/test_g2_survey")
    assert((survey6["pos"] == survey["pos"]).all())


def test_unit_persistence():
    f = pynbody.load("testdata/test_g2_survey")

    # f2 is the comparison case - just load the whole
    # position array and convert it, simple
    f2 = pynbody.load("testdata/test_g2_survey")
    f2['pos']
    f2.physical_units()

    f.gas['pos']
    f.physical_units()
    assert (f.gas['pos'] == f2.gas['pos']).all()

    # the following lazy-loads should lead to the data being
    # auto-converted
    f.dm['pos']
    assert (f.gas['pos'] == f2.gas['pos']).all()
    assert (f.dm['pos'] == f2.dm['pos']).all()

    # the final one is the tricky one because this will trigger
    # an array promotion and hence internally inconsistent units
    f.star['pos']

    assert (f.star['pos'] == f2.star['pos']).all()

    # also check it hasn't messed up the other bits of the array!
    assert (f.gas['pos'] == f2.gas['pos']).all()
    assert (f.dm['pos'] == f2.dm['pos']).all()

    assert (f['pos'] == f2['pos']).all()


def test_per_particle_loading():
    """Tests that loading one family at a time results in the
    same final array as loading all at once. There are a number of
    subtelties in the gadget handler that could mess this up by loading
    the wrong data."""

    f_all = pynbody.load("testdata/test_g2_survey")
    f_part = pynbody.load("testdata/test_g2_survey")

    f_part.dm['pos']
    f_part.star['pos']
    f_part.gas['pos']

    assert (f_all['pos'] == f_part['pos']).all()

