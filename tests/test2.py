from pprint import pprint
class Student:
    def __init__(self,name:str):
        self.name = name
        
    id:int = 1
    # name:str = "justin"
    # instance function
    def say_hello(self,msg:str):
        pprint("hhhhh")
    pass


def main():
    s1 = Student("justin")
    print(s1)
    print(hex(id(s1)))
    print(Student)
    print(isinstance(s1,Student))
    print(Student.__name__)
    print(Student.id)
    print(getattr(Student,"name"))
    print(getattr(Student,"unknown","no value!"))
    # Student.__setattr__
    Student.name = "jhon"
    print(Student.name)
    setattr(Student,"name","julite")
    print(Student.name)
    delattr(Student,"id")
    try:
        print(Student.id)
    except TypeError as e:
        print(str(e))
    except AttributeError as e:
        pprint(str(e))
    finally:
        pprint(Student.__dict__)
    
    s1.say_hello("你好？")
if __name__ == '__main__':
    main()