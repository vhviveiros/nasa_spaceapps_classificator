import multiprocessing
from job import job
import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor


def __process(images, masks):
    return (job(args) for args in np.swapaxes([images, masks], 0, 1))


def process(covid_images, covid_masks, non_covid_images, non_covid_masks):
    with ThreadPoolExecutor() as executor:
        cov_processed = executor.submit(
            __process, list(covid_images.result()), list(covid_masks.result()))
        non_cov_processed = executor.submit(
            __process, list(non_covid_images.result()), list(non_covid_masks.result()))

    return [cov_processed, non_cov_processed]
