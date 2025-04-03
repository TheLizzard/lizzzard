#lang racket

(define (fib x)
  (if (< x 1)
      1
      (+ (fib (- x 1)) (fib (- x 2)))))

(define n (string->number (vector-ref (current-command-line-arguments) 0)))

(displayln (format "fib(~a)=~a" n (fib n)))