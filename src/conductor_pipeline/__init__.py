from datetime import datetime
import luigi
import pandas as pd
from common import MSSqlConnection, ADLTarget


class ConductorPipeline(luigi.Task):
    """
    Main Pipeline class that conducts the kubernetes cluster to orchestrate 
    pipeline jobs for each course.
    """
    def requires(self):
        return EdxCourseIdsTask()

    def run(self):
        with self.input().open() as course_ids_file:
            edx_course_ids = pd.read_csv(course_ids_file)
        yield [SingleCourseKubernetesJobTask(course_id) for course_id in list(edx_course_ids)] 


class SingleCourseKubernetesJobTask(luigi.contrib.kubernetes.KubernetesJobTask):
    
    course_id = luigi.parameter()

    name = 'conductor'
    spec_schema = """{
        "containers": [{
            "name": "single_course",
            "image": "kabirkhan14:learner-attrition-single-course"
            "args": "{course_id}"
        }],
        "restartPolicy": "Never"
    }"""

    def run(self):
        self.spec_schema = self.spec_schema.format(course_id=self.course_id)
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
