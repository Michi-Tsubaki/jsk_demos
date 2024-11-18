#!/usr/bin/env roseus

;;Okada-seisei seminar
;;Surgery-method dev2
;;last editted Nov 12 ;; Michi-Tsubaki

;;<DESCRIPTION>
;;This program is for trying to make unique motion for sewing sponge. Still Trying

;;Load packages for using PR2 model and PR2 robot in real.
(require "package://pr2eus/pr2.l")
(require "package://pr2eus/pr2-utils.l")
(require "package://pr2eus/pr2-interface.l")
(require "package://pr2eus/speak.l") ;;pkg for speaking
(load "models/arrow-object.l")

;;Set PR2
(if (not (boundp '*pr2*)) (pr2-init)) 


