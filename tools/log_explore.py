#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

import sys
import re
from collections import OrderedDict

from shape_learning.shape_learner_manager import ShapeLearnerManager
from shape_learning.shape_learner import SettingsStruct
from shape_learning.shape_modeler import ShapeModeler

import os.path

from shape_learning.stroke import Stroke
from shape_learning import stroke
import argparse
parser = argparse.ArgumentParser(description='Learn a collection of letters in parallel')
parser.add_argument('word', action="store",
                help='The word to be learnt')
demo_letters = {}
robot_letters = {}

ref_for_6 = "[0.315939020957 0.236035640643 0.191334439377 0.156252159658 0.120406475039 0.0743485631942 0.0126352872995 -0.0619948980249 -0.132317993106 -0.186953171932 -0.231492890919 -0.302551989486 -0.349049309206 -0.385520018469 -0.411736719016 -0.438575926919 -0.465691566798 -0.481725958824 -0.491347356411 -0.5 -0.490726954099 -0.490474909783 -0.490226130364 -0.482566181155 -0.464961144233 -0.431012136204 -0.394727929049 -0.350535274752 -0.28849701313 -0.244230243211 -0.198913739324 -0.129098259972 -0.0548000954368 -0.00777711563358 0.0705139655534 0.110198034561 0.156665119164 0.190493290877 0.241841779869 0.297357897342 0.340394449212 0.386079149393 0.422109648711 0.466204181765 0.492689648912 0.499995642141 0.5 0.49671235835 0.472607807947 0.442823433012 0.416416913287 0.371382889593 0.336896463363 0.291816872107 0.229732486538 0.158578519528 0.0695004269382 0.00668631683082 -0.0579324956859 -0.14255964717 -0.197492504005 -0.249560601406 -0.276532572436 -0.293034278152 -0.320068711636 -0.356450771859 -0.392135030418 -0.420066469388 -0.454428133647 -0.49924029926 -0.938368454183 -0.911845754548 -0.884413720516 -0.857474260925 -0.832075131903 -0.82223824187 -0.786412178589 -0.732014419434 -0.673745351337 -0.613911634285 -0.558076744852 -0.479393129506 -0.388457259334 -0.334419616363 -0.272053095403 -0.190943331072 -0.0848451396242 0.0121980489563 0.110937392928 0.206590099032 0.308271559637 0.399241159535 0.482823825691 0.552637571408 0.58482151569 0.645443520785 0.697694721127 0.752469878661 0.806150886976 0.841902446381 0.868118841435 0.902559789023 0.911368404715 0.922915003727 0.932643635301 0.937409003232 0.938368454183 0.92928808078 0.911053691951 0.894098899842 0.868103600889 0.841313076203 0.796394182007 0.706960884772 0.643973583171 0.593874532328 0.527390561035 0.456584681804 0.366438178289 0.309364961661 0.242819132549 0.194976221594 0.151489976328 0.131339177777 0.105043047974 0.0958386569392 0.0960407389168 0.0869141452993 0.0974683585201 0.104275953124 0.13193486532 0.138741058986 0.147680407492 0.171911559113 0.183259020473 0.201010555731 0.234347200975 0.26039967989 0.30511837754 0.350140793901]"

ref_shape = literal_eval(ref_for_6.replace(' ',', '))
ref_shape = np.reshape(ref_shape, (-1, 1))
ref_stroke = Stroke()
ref_stroke.stroke_from_xxyy(np.reshape(ref_shape,len(ref_shape)))
ref_stroke.uniformize()

if __name__ == "__main__":

    with open(sys.argv[1], 'r') as log:
        boo = True
        letter = 6
        for line in log.readlines():

                if boo:
                    demo_letters.setdefault(letter,[]).append(line)
                    boo = not boo
                else:
                    robot_letters.setdefault(letter,[]).append(line)
                    boo = not boo

    robot_dists = []

    for letter, value  in robot_letters.items():

        for path in value:

            if len(path)>0:
                #path = path.strip('[')
                #path = path.strip(']\n')
                #path = "[" + str(path) + "]"
                userShape = literal_eval(path)
                userShape = np.reshape(userShape, (-1, 1))
                #print userShape
                demo_stroke = Stroke()
                demo_stroke.stroke_from_xxyy(np.reshape(userShape,len(userShape)))
                demo_stroke.uniformize()
                #print demo_stroke.x
                #print ref_stroke.x
                _,dist = stroke.euclidian_distance(demo_stroke,ref_stroke)
                robot_dists.append(dist)

    demo_dists = []

    for letter, value  in demo_letters.items():

        for path in value:

            if len(path)>0:
                #path = path.strip('[')
                #path = path.strip(']\n')
                #path = "[" + str(path) + "]"
                userShape = literal_eval(path)
                userShape = np.reshape(userShape, (-1, 1))
                #print userShape
                demo_stroke = Stroke()
                demo_stroke.stroke_from_xxyy(np.reshape(userShape,len(userShape)))
                demo_stroke.uniformize()
                #print demo_stroke.x
                #print ref_stroke.x
                _,dist = stroke.euclidian_distance(demo_stroke,ref_stroke)
                demo_dists.append(dist)


    plt.plot(robot_dists,'b')
    plt.plot(demo_dists,'r')
    plt.plot(robot_dists,'bs')
    plt.plot(demo_dists,'r.')
    plt.plot(np.ones(len(robot_dists))*0.4,'g')
    plt.show()
