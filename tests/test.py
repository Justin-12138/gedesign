num = input("enter a number:")
a=100
try:
    res = a/int(num)
    print(res)
except ValueError as e:
    print(f"error:{e}")
except ZeroDivisionError as e:
    print(e)

else:
    print("successfully!")


finally:
    print("end!")