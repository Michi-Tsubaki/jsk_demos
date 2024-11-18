#!/usr/bin/env roseus

;;Okada-sensei seminar
;;Surgery-method dev2
;;last editted Nov 18
;;editted by Michi-Tsubaki

;;<Description>
;;Trying to make unique motion for sewing sponge.


;;Load pkgs around PR2 Interface
(require "package://pr2eus/pr2.l")
(require "package://pr2eus/pr2-utils.l")
(require "package://pr2eus/pr2-interface.l") ;;*ri*
(require "package://pr2eus/speak.l") ;;pkg for speaking

;;Load class related
(load "trajectory.l")
(load "initial-task.l")
(load "env.l")
(load "models/arrow-object.l") ;;*arrow*

;;Set arrow (for debug)
(setq *arrow* (arrow))
(send *arrow* :copy-worldcoords)

;;Set remain (TBU: to be used)
(setq *remain* 1000) ;;how long remaining thread?

;;Set PR2
(if (not (boundp '*pr2*)) (pr2-init)) 

;;Set environment parameter
(setq *centerx* 700) ;;change this param
(setq *centery* 0)  ;;change this param
(setq *centerz* 750)  ;;change this param
(setq *deskw* 500) ;;change this param
(setq *deskh* 750) ;;change this param

;;Set Experiment environment
(set-env) ;; check env.l


;;Show all objects in IRTVIEWER
(objects (list *pr2* *center* *arrow* *desk* *needle* *hampen*))


;;Human help Robot to grab needle
(initial-task) ;; check initial-task.l




;;Make Trajectory 
(setq r (instance traj :init)) ;;check trajectory.l
(send r :rotate pi/2 :z)
(setq *target* (v+ (send *center* :pos) #f(0 10 30)))
(send r :locate *target*  :world)

(objects (append (list *pr2* *center* *arrow* *desk* *needle* *hampen*) (send r :points)))



;;Set Start Position
;;Larm
(setq current-coords (send *o* :copy-worldcoords))
(setq new-coords (send (send current-coords :translate (float-vector 0 200 200)) :rotate (deg2rad 90) :z))
(send new-coords :draw-on :flush t)

(send *pr2* :larm :inverse-kinematics new-coords :rotation-axis :y)
(send *pr2* :larm :end-coords :assoc *needle*)
(send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)

;;Rarm
(setq current-coords (send *o* :copy-worldcoords))
(setq new-coords (send current-coords :translate (float-vector 0 -100 500))) 
(send *pr2* :rarm :inverse-kinematics new-coords :rotation-axis :x)
(send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)


(dolist (e (send r :points))
  (send *pr2* :larm :inverse-kinematics e
	:rotation-axis :y
	:debug-view t)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  (send *irtviewer* :draw-objects)
  (format t "debug~%")
  (unix:sleep 2)
  
  )
