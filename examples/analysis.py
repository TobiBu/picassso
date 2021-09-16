"""

example analysis
================

An example of analysis flow using the picasso framework.
Here we examplify how to read in a survey, fit the images and the use
the resulting geometry of the galaxies to instantiate isophote objects.
From there we can calculate property totals in different radii bins and
perform gradient fits...

"""

import picassso as p
from picassso.analysis import util


s = p.load('./')

true, pred, dm = util.get_predicted_vs_true_data(s, filename='predicted_vs_true.h5', plot=False)


