#Solution taken from https://github.com/KarlosTan
# https://github.com/rohitash-chandra/python-bootcamp/tree/main/exercise1/part1/solution


from datetime import date
import numpy as np

def find_age():
    birthinput = input('Please enter your date of birth (DDMMYYYY): ')
    if len(birthinput) != 8:
        print('Entered date of birth invalid')
        find_age()
    else:
        today = date.today().strftime('%Y%m%d')
        birthdate = birthinput[4:9]+birthinput[2:4]+birthinput[0:2]             #Use YYYYMMDD to prioritise year, then month, then day
        age = int((int(today)-int(birthdate))/10000)                            #eg 20210101 - 20200101 = 10000 > 0 so 20210101 is a later date
        print('Your age is ', age, '.', end = ' ')
        if age > 18:
            print('Using adult BMI classification.')
        else:
            print('Using child BMI classification.')
        return age

def find_height():
    print('Would you like to input your height in metres, or feet and inches?')
    unit = input('Enter m for metres or f for feet and inches here: ')                                               #Ask for unit of input   
    if unit == 'm':
        height = float(input('Please enter your height in metres up to two decimal places. (eg. 1.78): '))
    elif unit == 'f':
        heightinput = input('Please enter your height in feet and nearest inch, separated by \'. (eg. 5\'7): ')      #Just going to trust they entered it correctly
        feet_inch = heightinput.split(sep = '\'')                                                                    #Find the ' and split the numbers into 2
        height = float(feet_inch[0])*0.3048 + float(feet_inch[1])*0.0254
    else:
        find_height()
    return height

def find_weight():
    weight = float(input('Please enter your weight in kg. (eg. 53): '))                                             #Also going to trust they entered a number
    return weight

def calc_BMI(weight, height):
    BMI = weight/height**2
    print('Your Body Mass Index is ', BMI)
    return BMI

def calc_class(BMI, age):
    #adult_classes = np.array([16.00, 17.00, 18.50, 25.00, 30.00, 35.00, 40.00])             #Use elementwise comparison then count the number of True
    #adult_groups = ['underweight -severe thinness','underweight - moderate thinness', 'underweight- mild thinness','normal', 'overweight- pre-obese', 'obese - class 1', 'obese - class 2', 'obese - class 3']
    #child_classes = np.array([20.00, 29.00, 32.00])
    #child_groups =['underweight','heavy weight','overweight', 'obese' ]

    if age > 18:
        if BMI < 16:
            print('underweight -severe thinness')
        elif BMI >16 and BMI <=17:
            print('underweight -moderate thinness')
        elif BMI >17 and BMI <=18.5:
            print('underweight -mild thinness')
        #complete the rest 
    else:
        if BMI < 20:
            print('underweight')
        elif BMI >20 and BMI <=29:
            print('heavy weight')
        #complete the rest 


def main(): 
    quit = input('Press the enter key to start or enter q to quit: ')
    if quit != 'q':
        print('BMI Calculator')
        age = find_age()
        height = find_height()
        weight = find_weight()
        BMI = calc_BMI(weight, height)
        print(BMI, ' is your BMI. Next we find your group')
        calc_class(BMI, age)
        main()

if __name__ == '__main__':
    main()
