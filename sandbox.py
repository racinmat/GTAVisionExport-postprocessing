import pickle

from extract_consecutive_screenshots import get_pickle_name, load_objects
from visualization import get_connection

if __name__ == '__main__':
    conn = get_connection()
    run_id = 17
    objects = load_objects(run_id)
    pass

