#lang racket

(define (f x)
  (if (> x 0)
      (+ (f (- x 1)) 1)
      0))

(define n (string->number (vector-ref (current-command-line-arguments) 0)))

(displayln (f n))