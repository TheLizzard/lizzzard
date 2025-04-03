#lang racket

(define (main n)
  (define flags (make-vector n #t))
  (define count 0)
  
  (for ([i (in-range 2 n)])
    (when (vector-ref flags i)
      (let loop ([j (* 2 i)])
        (when (< j n)
          (vector-set! flags j #f)
          (loop (+ j i))))
      (set! count (+ count 1))))
  
  (displayln (string-append "Primes up to " (number->string n) " " (number->string count))))

;; Read integer from command-line argument
(define n (string->number (vector-ref (current-command-line-arguments) 0)))
(main n)