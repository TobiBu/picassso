# configurations for picasso

from pathlib import Path

def _get_basic_config():

    config = {'verbose': False}

    return config


def configure_physical_scaling():

    config['scale_to_physical'] = True
    

def configure_directories():
	# set paths based on home directory:

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

    elif user == "buck":
        path_prefix = f'/Users/buck/Documents/PhD/machine_learning/data/'
        path_predicted = f'/Users/buck/Documents/PhD/machine_learning/data//val_100000/' #'/isaac/ptmp/gc/tbuck/PhD_preparation/Machine_learning/data/truth/'
        path_true = f'/Users/buck/Documents/PhD/machine_learning/data/val_100000/true/' 
        #f'/isaac/ptmp/gc/tbuck/PhD_preparation/Machine_learning/data/truth/' #'/isaac/ptmp/gc/tbuck/PhD_preparation/Machine_learning/data/prediction/val_100000/true/' #'/isaac/ptmp/gc/tbuck/PhD_preparation/Machine_learning/data/truth/'
        pvt_file = path_prefix + 'predicted_vs_true.h5'
        dm_file = path_prefix + 'illustris_fof_props.h5'
    else:
        print("unknown user")
    
    config['path_prefix'] = path_prefix
    config['path_predicted'] = path_predicted
    config['path_true'] = path_true
    config['pvt_file'] = pvt_file
    config['dm_file'] = dm_file
 
    config['families'] = {'train':"t, tr",'prediction':"p, pred, predict","validation":"val, v, valid"}

    config['survey-class-priority'] = ['SDSSMockSurvey']#,'ClassicSurvey','MANGASurvey']


def configure_survey_loading_priority():
    from . import survey

    # Turn the config strings for survey classes into lists of
    # actual classes
    _survey_classes_dict = dict([(x.__name__, x) for x in survey._get_survey_classes()])
    config['survey-class-priority'] = [_survey_classes_dict[x]
                                     for x in config['survey-class-priority']]


config = _get_basic_config()
configure_physical_scaling()
configure_directories()
configure_survey_loading_priority()
