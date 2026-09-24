from pathlib import Path
import sys
import unittest
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flight_evaluator import FlightEvaluator, EvaluationState


class EvaluatorTests(unittest.TestCase):
    def evaluator(self):
        return FlightEvaluator([160,0,-6], [[-20,-20,-20],[200,20,0]], 0, 160)

    def state(self, t, speed=0, collided=False, position=(160,0,-6)):
        return EvaluationState(t,position,(speed,0,0),collided)

    def test_arrival_needs_stop_request_and_unbroken_dwell_below_limit(self):
        e = self.evaluator()
        self.assertIsNone(e.update(self.state(1),False))
        self.assertIsNone(e.update(self.state(2),True))
        self.assertIsNone(e.update(self.state(2.5,speed=.5),True))
        self.assertIsNone(e.update(self.state(3),True))
        self.assertIsNone(e.update(self.state(3.9),True))
        self.assertTrue(e.update(self.state(4),True)['success'])

    def test_collision_takes_precedence_over_success(self):
        e = self.evaluator()
        e.update(self.state(1),True)
        self.assertEqual(e.update(self.state(2,collided=True),True)['termination'],'collision')

    def test_goal_boundary_and_timeout_are_not_success(self):
        e = self.evaluator()
        self.assertIsNone(e.update(self.state(1,position=(160,0,-9)),True))
        self.assertEqual(e.update(self.state(180),True)['termination'],'timeout')
        e = self.evaluator()
        self.assertEqual(e.update(self.state(1,position=(160,21,-6)),True)['termination'],'boundary_exit')

    def test_path_efficiency_is_bounded_even_when_reference_is_longer(self):
        e = self.evaluator()
        e.update(self.state(1,position=(158,0,-6)),False)
        e.update(self.state(2),True)
        result = e.update(self.state(3),True)
        self.assertEqual(result['path_efficiency'],1.)
        json.dumps(result,allow_nan=False)

    def test_result_normalizes_external_boolean_scalars(self):
        class ExternalBoolean:
            def __bool__(self): return True
        e = self.evaluator()
        e.update(self.state(1),ExternalBoolean())
        result=e.update(self.state(2),ExternalBoolean())
        self.assertIs(result['stop_requested'],True)
        json.dumps(result,allow_nan=False)


if __name__ == '__main__':
    unittest.main()
