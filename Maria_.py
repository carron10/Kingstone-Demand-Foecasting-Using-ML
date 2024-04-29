# %% [markdown]
# Maria Dongeri |
# N02212150N |
# Acturial sciences

# %%
input_age = input("Enter dog Age:")
try:
    dog_age = float(input_age)
    if dog_age < 0:
        print("Age cannot be a negative number.")
    else:
        human_age = None
        if dog_age <= 1:
            human_age = 15*dog_age
        elif dog_age <= 2:
            human_age = dog_age * 12
        elif dog_age <= 3:
            human_age = 9.3 * dog_age
        elif dog_age <= 4:
            human_age = 8 * dog_age
        elif dog_age <= 5:
            human_age = 7.2 * dog_age
        else:
            human_age = (7.2 * 5) + (7*(dog_age-5))
        dog_age,human_age=round(dog_age,2),round(human_age,2)
        print(f"The given dog age {dog_age} is {human_age} in human years.")
except Exception as e:
    print(f"{input_age} is Invalid")


