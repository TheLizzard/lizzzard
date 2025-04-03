#lang racket

(define A (make-hash))
(hash-set! A 'X 0) ; Define an object with attribute X

(define n (string->number (vector-ref (current-command-line-arguments) 0)))

(define (increment-X)
  (when (< (hash-ref A 'X) n)
    (hash-set! A 'X (+ (hash-ref A 'X) 1))
    (increment-X)))

(increment-X)

(displayln (hash-ref A 'X))