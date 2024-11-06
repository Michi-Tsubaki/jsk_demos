#!/usr/bin/env roseus
;;Okada-seisei seminar
;;Surgery-robot
;;last editted Nov 6 ;; Michi-Tsubaki

;;<DESCRIPTION>
;;This program is for trying to make unique motion for sewing sponge. Still Trying


;;Load packages for using PR2 model and PR2 robot in real.
(require "package://pr2eus/pr2.l")
(require "package://pr2eus/pr2-utils.l")
(require "package://pr2eus/pr2-interface.l")
(require "package://pr2eus/speak.l")
(load "models/arrow-object.l")


;;Make PR2 object
(if (not (boundp '*pr2*)) (pr2-init)) 


;;Set Center of a Table
(setq *center* (make-cube 10 10 10))
(send *center* :translate (float-vector 700 0 750))
(send *center* :set-color :black)

;;Setting cube as a desk for surgery. 
(setq *desk* (make-cube 500 500 750))
(send *desk* :translate (float-vector 700 0 750/2)) 
(send *desk* :set-color :brown)


;;Setting cylinder as needle.
(setq *needle* (make-cylinder 1 40)) 
(send *needle* :translate (v+ (float-vector 0 20 100) (send *center* :pos)))
(send *needle* :set-color :yellow)
(send *needle* :rotate #d90 :x)

;;Set hampen
;(setq *hampen* (make-cube 75 120 30))
;(send *hampen* :translate (v+ (float-vector 0 0 30/2) (send *center* :pos)))
;(send *hampen* :set-color :white)

;;Set left hampen
(setq *left-hampen* (make-cube 75 60 30))
(send *left-hampen* :translate (v+ (float-vector 0 -32 30/2) (send *center* :pos)))
(send *left-hampen* :set-color :white)

(setq *right-hampen* (make-cube 75 60 30))
(send *right-hampen* :translate (v+ (float-vector 0 32 30/2) (send *center* :pos)))
(send *right-hampen* :set-color :white)


;;Set coordination.
(send *desk* :put :left-coords
      (make-cascoords
       :coords (send (send *desk* :copy-worldcoords) :translate (float-vector 0 0 500))
       :parent *pr2*))
(send *needle* :put :left-coords
      (make-cascoords
       :coords (send (send *needle* :copy-worldcoords) :translate (float-vector 0 0 -5))
       :parent *needle*)) ;;質問：parent座標をpr2に設定したら収束しなくなった ;;still confusing 

;;Show Worldrecord
(setq *arrow* (arrow))
(send *arrow* :copy-worldcoords)

;;Show all objects in viewer. 
(objects (list *pr2* *center* *arrow* *desk* *needle*  *left-hampen* *right-hampen*)) 


;Set initial pose
(send *ri* :speak-jp "初期姿勢に戻ります。" :wait t)
(send *pr2* :reset-pose)
(send *ri* :start-grasp :arms)
(send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)

(send *ri* :speak-jp "右手を開きます。スポンジをもたせてください。" :wait t)
(send *ri* :stop-grasp :rarm :wait t)
(send *irtviewer* :draw-objects)

(send *ri* :speak-jp "閉じます。" :wait t)
(send *ri* :start-grasp :rarm :wait t)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)

(send *ri* :speak-jp "左手を開きます。針を持たせてください。")
(send *ri* :stop-grasp :larm :wait t)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)

(send *ri* :speak-jp "閉じます。" :wait t)
(send *ri* :start-grasp :larm :wait t)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)

(send *ri* :speak-jp "これから針を持って作業します．離れてください．" :wait t)
(unix:sleep 2)

(send *pr2* :larm :inverse-kinematics
      (send (send *desk* :get :left-coords) :copy-worldcoords)
      :rotation-axis :z)
;(send *pr2* :rarm :inverse-kinematics
;      (send (send *desk* :get :left-coords) :copy-worldcoords)
;      :rotation-axis :z)
(send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
(send *ri* :wait-interpolation)
(send *irtviewer* :draw-objects)
;(unix:sleep 1)
