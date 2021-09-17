import picasso as p
from picasso.analysis import util

bands = 'r'

s = p.load('./',  0, bands)
true, pred, dm = util.get_predicted_vs_true_data(s, filename='predicted_vs_true_'+bands+'.h5', plot=False)
