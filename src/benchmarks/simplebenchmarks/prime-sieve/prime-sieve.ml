(* The filter function takes a generator (a function unit -> int) and a prime,
   and returns a new generator that skips values divisible by that prime. *)
let filter gen prime =
  fun () ->
    let rec aux () =
      let x = gen () in
      if x mod prime <> 0 then x else aux ()
    in
    aux ()

(* The base generator yields the natural numbers starting from 2 *)
let base_generator () =
  let i = ref 2 in
  fun () ->
    let r = !i in
    incr i;
    r

(* The main function computes the nth prime by repeatedly updating the generator *)
let rec main n gen =
  if n <= 0 then invalid_arg "n must be positive"
  else
    let prime = gen () in
    if n = 1 then
      Printf.printf "%d\n" prime
    else
      main (n - 1) (filter gen prime)

(* Entry point: reads n from command-line arguments and runs main *)
let () =
  if Array.length Sys.argv <> 2 then
    Printf.printf "Usage: %s <n>\n" Sys.argv.(0)
  else
    let n = int_of_string Sys.argv.(1) in
    main n (base_generator ())