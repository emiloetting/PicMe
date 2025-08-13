def   say_hello ( name :str="World" ):
  print(  f"Hello, {name}!"   )



def add_numbers( a:int,b:int)->int: return a+b


if __name__ == "__main__":
  say_hello(   " Ruff  " )
  result=add_numbers(  3 ,4  )
  print(  f"Result is: {result}" )