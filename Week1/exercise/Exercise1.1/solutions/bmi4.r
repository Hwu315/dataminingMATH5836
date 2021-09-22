
#source: http://applied-r.com/conditionals-in-r/
calc_bmi <- function(height, weight){
    bmi = round(weight/(height**2),2)
    #print(bmi)
    return(bmi)
}

bmi_func <- function(height, weight){
    BMI = calc_bmi(height, weight)
 
    if (BMI < 18.5)
        print(" Underweight")
    else if ((BMI >= 18.5) && (BMI <= 24.9)){
        print("Healthy weight range")}
    else if ((BMI >= 25) && (BMI <= 29.9)){
        print(" Overweight")}
    else
       print(" Obese ")

    return(BMI)

}

# Test the function  
mybmi= calc_bmi(1.78, 99)
print(mybmi)

 
print("Calculation of your Body Mass Index (BMI)") 

cat("Enter height in meters: ")
height <- readLines("stdin", 1)
height <- as.numeric(height)


cat("Enter weight in kg: ")
weight <- readLines("stdin", 1)
weight <- as.numeric(weight)

#height <- readline(prompt="Enter height in meters: ") 
#height <- as.numeric(height) 
#weight <- readline(prompt="Enter weight in kg: ") 
#weight <- as.numeric(weight) 

bmi = bmi_func(height, weight)
print('Your BMI is: ')
print(bmi)

