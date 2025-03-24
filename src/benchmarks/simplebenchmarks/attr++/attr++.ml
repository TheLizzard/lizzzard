type a = { mutable x : int }

let () =
  let n = int_of_string Sys.argv.(1) in
  let a = { x = 0 } in
  while a.x < n do
    a.x <- a.x + 1
  done;
  Printf.printf "%d\n" a.x