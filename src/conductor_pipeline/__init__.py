from datetime import datetime
import luigi
from luigi.contrib.kubernetes import KubernetesJobTask
import pandas as pd
from common import MSSqlConnection, ADLTarget


class ConductorPipeline(luigi.Task):
    """
    Main Pipeline class that conducts the kubernetes cluster to orchestrate 
    pipeline jobs for each course.
    """
    # def requires(self):
    #     return EdxCourseIdsTask()

    def complete(self):
        return False

    def run(self):
        # with self.input().open() as course_ids_file:
        #     edx_course_ids = pd.read_csv(course_ids_file)
        yield SingleCourseKubernetesJobTask('Microsoft+DAT222x+4T2017')
        # yield [SingleCourseKubernetesJobTask(course_id) for course_id in list(edx_course_ids)] 


class SingleCourseKubernetesJobTask(KubernetesJobTask):
    
    course_id = luigi.Parameter()

    name = 'conductor'
    spec_schema = {
        "containers": [{
            "name": "single_course",
            "image": "learnerattrition.azurecr.io/learner-attrition"
        }],
        "imagePullSecrets": [{
            "name": "registrykey"
        }]
    }

    def run(self):
        print('COURSE ID: ', self.course_id)
        self.spec_schema["containers"][0]["args"] = [self.course_id]

        print('=====================================================')
        print('=====================================================')
        print(self.spec_schema)
        print('=====================================================')
        print('=====================================================')
        super().run()
    

class EdxCourseIdsTask(luigi.Task):
    """
    Get all edx course ids to run pipelines for
    """
    _query = """
        SELECT DISTINCT CourseRunId
        FROM [EdxDW].[edx].[DimCourse] C
        WHERE [CourseRunStartDate] < {current_date}
        AND [CourseRunEndDate] > {current_date}
    """

    def output(self):
        return ADLTarget('data/edx_course_ids.csv')

    def run(self):
        current_date_string = datetime.strftime(datetime.today(), '%Y-%m-%d')

        conn = MSSqlConnection()
        edx_course_ids = conn.run_query(self._query.format(current_date=current_date_string))
        edx_course_ids.columns = ['edx_course_id']
        
        with self.output().open('w') as output:
            edx_course_ids.to_csv(output, index=False)


if __name__ == "__main__":
    luigi.run()
