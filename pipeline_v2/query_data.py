# -*- coding: utf-8 -*-

import luigi


class RawUserActivity(luigi.ExternalTask):
    """
    Tasks that implement this mixin must return data in a specific format
    for events, course_starts, course_completions, forum posts and course start/end dates
    """

    course_id = luigi.Parameter()

    RAW_DATA_FILE_NAMES = [
        'events',
        'forums',
        'course_starts',
        'course_completions',
        'course_dates'
    ]

    sql_query = ""

    def output(self):
        out = {}
        for name in self.RAW_DATA_FILE_NAMES:
            out[name] = luigi.LocalTarget('data/{course_id}/{name}.csv'.format(
                course_id=self.course_id, name=name
            ))

        return out


if __name__ == "__main__":
    luigi.run()