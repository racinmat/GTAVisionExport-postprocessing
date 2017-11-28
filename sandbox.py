import pickle

from extract_consecutive_screenshots import get_pickle_name, load_objects
from visualization import get_connection

if __name__ == '__main__':
    conn = get_connection()
    run_id = 11
    cars = load_objects(run_id)
    car1 = cars[cars.keys()[0]]
    pass
