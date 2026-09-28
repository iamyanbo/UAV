"""Label-side observable expert. Never import into inference.

Privileged geometry supplies visibility/clearance labels only. An unseen goal
never supplies a steering direction: unresolved search uses an inspection turn.
"""
import math
import numpy as np


def observable_expert(position, heading, velocity, goal, goal_heading, calibration, field, quaternion=None):
    c,s=math.cos(heading),math.sin(heading)
    body_to_world=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    camera_body_to_world=body_to_world
    if quaternion is not None:
        x,y,z,w=np.asarray(quaternion)/np.linalg.norm(quaternion)
        camera_body_to_world=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    camera_to_body=np.asarray(calibration['camera_to_body_rotation']).reshape(3,3)
    origin=np.asarray(calibration['camera_origin_body_m'])
    def visible(point):
        body=camera_body_to_world.T@(np.asarray(point)-position)
        camera=camera_to_body.T@(body-origin)
        if camera[2]<=.1:return False
        u=calibration['fx']*camera[0]/camera[2]+calibration['cx']
        v=calibration['fy']*camera[1]/camera[2]+calibration['cy']
        return bool(0<=u<640 and 0<=v<480 and not field.swept_collision(position,point))
    goal=np.asarray(goal);seen=visible(goal)
    delta=body_to_world.T@(goal-position)
    heading_error=(goal_heading-heading+math.pi)%(2*math.pi)-math.pi
    near=np.linalg.norm(delta[:2])<=3 and abs(delta[2])<=2
    seen=seen or bool(near and abs(heading_error)<=math.radians(30) and not field.swept_collision(position,goal))
    arrival=bool(near and abs(heading_error)<=math.radians(30) and np.linalg.norm(velocity)<.5)
    command=[0.,0.,0.,15.];intention='search';altitude='maintain'
    if not seen and all(visible(position+body_to_world@np.array([distance,side,down]))
            for distance in (2.,3.,4.) for side,down in ((0,0),(.75,0),(-.75,0),(0,.75),(0,-.75))):
        command=[1.5,0.,0.,0.]  # Local visible-space exploration, no hidden-goal bearing.
    if seen:
        intention='approach';angle=math.atan2(delta[1],delta[0])
        command=[0.,0.,0.,float(np.clip(math.degrees(angle)*1.5,-45,45))]
        if near:
            command=[0.,0.,0.,float(np.clip(math.degrees(heading_error)*1.5,-45,45))]
        elif abs(angle)<math.radians(22.5):
            command[0]=min(2.5,float(np.linalg.norm(delta[:2])))
            command[2]=float(np.clip(delta[2],-.5,.5))
            # Entire sampled motion corridor must have visible, clear support.
            supported=all(visible(position+body_to_world@(np.asarray(command[:3])*t+offset))
                for t in (.5,1.,2.) for offset in (np.zeros(3),np.array([0,.75,0]),
                    np.array([0,-.75,0]),np.array([0,0,.75]),np.array([0,0,-.75])))
            if not supported:command[:3]=[0.,0.,0.]
        altitude='gain' if command[2]<-.1 else 'lose' if command[2]>.1 else 'maintain'
    if arrival:command=[0.]*4;intention='hold'
    return dict(schema='observable-expert/v1',teacher_command=command,intention=intention,altitude=altitude,
                policy_valid=False,eligibility_reason='geometry does not establish visual recognizability or an observable search decision',
                goal_visibility=float(seen),goal_evidence=float(seen)*math.exp(-float(np.linalg.norm(delta))/20.),
                arrival=arrival,independent=True,grounding='goal_view_agreement_label' if arrival else
                'visible_goal_and_corridor' if seen else 'observable_local_search')
