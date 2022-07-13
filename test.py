class Dog:
    d_t = "金毛"  # 类属性，类变量，公共属性

    def say_hi(self):  # self代表实例本身
        print("my type is", self.d_t)


d = Dog()  # 生成一个实例
d.say_hi()  # 调用实例化方法
