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
(setq *centerz* 800)  ;;change this param
(setq *deskw* 500) ;;change this param
(setq *deskh* *centerz*) ;;change this param

;;Set Experiment environment
(set-env) ;; check env.l


;;Show all objects in IRTVIEWER
(objects (list *pr2* *center* *arrow* *desk* *needle* *hampen*))


;;Human help Robot to grab needle
(initial-task) ;; check initial-task.l


    ;;Grasping needle (dummy)
(send *pr2* :larm :inverse-kinematics
      (send (send (send *needle* :get :left-coords)
		  :copy-worldcoords)
	    :rotate (deg2rad 90) :z)
      :rotation-axis :z
      :debug-view t)
(send *pr2* :larm :end-coords :assoc *needle*)
(send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)


;;Make Trajectory 
(setq r (instance traj :init)) ;;check trajectory.l
(send r :rotate pi/2 :z)
(setq *target* (v+ (send *center* :pos) #f(0 40 30)))
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
(setq *r-pos* (send (elt (send r :points) 9) :copy-worldcoords))

(send *r-pos* :rotate pi/2 :y :local)
(send *r-pos* :rotate -pi/2 :z :local)
(send *r-pos* :translate #f(-10 0 0) :local)
;;(setq current-coords (send *o* :copy-worldcoords))
;;(setq new-coords (send current-coords :translate (float-vector 0 -100 500))) 
(send *pr2* :rarm :inverse-kinematics *r-pos*  :rotation-axis t
      :debug-view t
      :use-torso t)
(send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
(send *ri* :stop-grasp :rarm)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)


(dolist (e (send r :points))
  (let (ee)
    (setq ee (send e :copy-worldcoords))
    (send ee :translate #f(68 0 0) :local)
    (send ee :rotate pi :z :local)
    (send ee :rotate pi/2 :y :local)
    (send *pr2* :larm :inverse-kinematics ee
	  :rotation-axis t
	  :debug-view t)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  (send *irtviewer* :draw-objects)
  (format t "debug~%")
  (unix:sleep 2)
  )
  )


;;
(send *ri* :start-grasp :rarm :wait t)
(send *ri* :wait-interpolation)

;;dessoc needle assoc needle
(send *pr2* :larm :end-coords :dissoc *needle*)
(send *pr2* :rarm :end-coords :assoc *needle*)
(send *ri* :stop-grasp :larm :wait t)
(send *ri* :wait-interpolation)

;;draw needle
(send *pr2* :rarm :move-end-pos #f(0 0 -100) :local)
(send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
(send *ri* :wait-interpolation) 
(send *irtviewer* :draw-objects)

;;switch pos larm
(setq *o* (send *center* :copy-worldcoords))
(setq *pass-larm* (send *o* :translate #f(0 35 200)))
(send *pass-larm* :rotate pi/2 :z)
(send *pass-larm* :rotate pi/2 :y)
(Send *pass-larm* :rotate pi :x)
(send *pass-larm* :draw-on :flush t :size 100)
(send *pr2* :larm :inverse-kinematics *pass-larm*  :rotation-axis t
      :use-torso t)
(send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)

;;switch pos rarm
(setq *o* (send *center* :copy-worldcoords))
(setq *pass-rarm* (send *o* :translate #f(0 -35 200)))
(send *pass-rarm* :rotate -pi/2 :x)
(send *pass-rarm* :translate #f(-10 0 0) :local)
(send *pass-rarm* :draw-on :flush t :size 100)
(send *pr2* :rarm :inverse-kinematics *pass-rarm*  :rotation-axis t)
(send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)


(send *ri* :start-grasp :larm)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)

(send *ri* :stop-grasp :rarm)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)


(send *pr2* :rarm :end-coords :dissoc *needle*)
(send *pr2* :larm :end-coords :assoc *needle*)




;;Rarm
(setq *r-pos* (send (elt (send r :points) 9) :copy-worldcoords))

(send *r-pos* :rotate pi/2 :y :local)
(send *r-pos* :rotate -pi/2 :z :local)
(send *r-pos* :translate #f(-10 0 0) :local)
;;(setq current-coords (send *o* :copy-worldcoords))
;;(setq new-coords (send current-coords :translate (float-vector 0 -100 500))) 
(send *pr2* :rarm :inverse-kinematics *r-pos*  :rotation-axis t
      :debug-view t
      :use-torso t)
(send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
(send *ri* :stop-grasp :rarm)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)


(dolist (e (send r :points))
  (let (ee)
    (setq ee (send e :copy-worldcoords))
    (send ee :translate #f(68 0 0) :local)
    (send ee :rotate pi :z :local)
    (send ee :rotate pi/2 :y :local)
    (send *pr2* :larm :inverse-kinematics ee
	  :rotation-axis t
	  :debug-view t)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  (send *irtviewer* :draw-objects)
  (format t "debug~%")
  (unix:sleep 2)
  )
  )


;;
(send *ri* :start-grasp :rarm :wait t)
(send *ri* :wait-interpolation)

;;dessoc needle assoc needle
(send *pr2* :larm :end-coords :dissoc *needle*)
(send *pr2* :rarm :end-coords :assoc *needle*)
(send *ri* :stop-grasp :larm :wait t)
(send *ri* :wait-interpolation)

;;draw needle
(send *pr2* :rarm :move-end-pos #f(0 0 -100) :local)
(send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
(send *ri* :wait-interpolation) 
(send *irtviewer* :draw-objects)

;;switch pos larm
(setq *o* (send *center* :copy-worldcoords))
(setq *pass-larm* (send *o* :translate #f(0 35 200)))
(send *pass-larm* :rotate pi/2 :z)
(send *pass-larm* :rotate pi/2 :y)
(Send *pass-larm* :rotate pi :x)
(send *pass-larm* :draw-on :flush t :size 100)
(send *pr2* :larm :inverse-kinematics *pass-larm*  :rotation-axis t
      :use-torso t)
(send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)

;;switch pos rarm
(setq *o* (send *center* :copy-worldcoords))
(setq *pass-rarm* (send *o* :translate #f(0 -35 200)))
(send *pass-rarm* :rotate -pi/2 :x)
(send *pass-rarm* :translate #f(-10 0 0) :local)
(send *pass-rarm* :draw-on :flush t :size 100)
(send *pr2* :rarm :inverse-kinematics *pass-rarm*  :rotation-axis t)
(send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)


(send *ri* :start-grasp :larm)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)

(send *ri* :stop-grasp :rarm)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)


(send *pr2* :rarm :end-coords :dissoc *needle*)
(send *pr2* :larm :end-coords :assoc *needle*)
