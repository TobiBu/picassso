# configurations for picasso

def _get_basic_config():

    config = {'verbose': False}

    return config


def configure_physical_scaling():

    config['scale_to_physical'] = True

def configure_directories():

	# set paths based on home directory:

	from pathlib import Path

    user = Path.home().name

    if user == "tbuck":
        path_prefix = f'/isaac/ptmp/gc/tbuck/PhD_preparation/Machine_learning/data/'
        path_predicted = f'/isaac/ptmp/gc/tbuck/PhD_preparation/Machine_learning/data/prediction/val_100000/' #'/isaac/ptmp/gc/tbuck/PhD_preparation/Machine_learning/data/truth/'
        path_true = f'/isaac/ptmp/gc/tbuck/PhD_preparation/Machine_learning/data/prediction/val_100000/true/' 
#f'/isaac/ptmp/gc/tbuck/PhD_preparation/Machine_learning/data/truth/' #'/isaac/ptmp/gc/tbuck/PhD_preparation/Machine_learning/data/prediction/val_100000/true/' #'/isaac/ptmp/gc/tbuck/PhD_preparation/Machine_learning/data/truth/'
        pvt_file = path_prefix + 'predicted_vs_true.h5'
        dm_file = path_prefix + 'illustris_fof_props.h5'
        
    elif user == "swolf":
        path_prefix = '/mnt/data1/swolf/deepspacelearning_experiments/sparseloss_16/prediction/val_100000/'
        path_predicted = f'/mnt/data1/swolf/deepspacelearning_experiments/sparseloss_16/prediction/val_100000/'
        path_true = f'/mnt/data1/swolf/deepspacelearning_experiments/sparseloss_16/prediction/val_100000/true/'
        pvt_file = path_prefix + 'predicted_vs_true.h5'
        dm_file = '/mnt/data1/swolf/deepspacelearning_experiments/illustris_fof_props.h5'

    else:
        print("unknown user")
    
    config['path_prefix'] = path_prefix
    config['path_predicted'] = path_predicted
    config['path_true'] = path_true
    config['pvt_file'] = pvt_file
    config['dm_file'] = dm_file
 
    config['families'] = {'train':"t, tr",'prediction':"p, pred, predict","validation":"val, v, valid"}

    config['survey-class-priority'] = ['SDSSMockSurvey','ClassicSurvey','MANGASurvey']

    config['verbose'] = False

config = _get_basic_config()
