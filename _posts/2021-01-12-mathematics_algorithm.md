---
layout: single
header:
  overlay_color: "#333"
  teaser: /assets/images/algorithm/img.jpg
title:  "Basic Mathematical Algorithm Used in Programming "
excerpt: "Data Structure & Algorithm"
breadcrumbs: true
share: true
permalink: /programming/
date:    2020-01-12 
toc: false

---
### Hey! folks this is a approach to get started with the simple `mathematical algorithm` using python starting with the basic one.

<h4 style="color:orange;">Check if number is palindrome</h4>

```python
# Palindrome numaber finding ie. INPUT - 121 and OUTPUT - 121 then number is palindrome
# The time complexity for this solution will be O(d) where d is number of digit
def palindrome(n):
    temp=n  
    rev=0
    while(n>0):
        z=n%10         
        rev=rev*10 + z
        n=int(n/10)
    return (temp==rev)
```
<h4 style="color:orange;">Count number of digit in a number</h4>

```python
# count the digit in a number eg. INPUT - 1234 OUTPUT - 4
# The time complexity for this solution is O(d) where d is number of digit  
def count1(n):      
    count=0
    while(n>0):
        count+=1
        n=int(n/10)
    return count
# Time complexity for this solution is O(1) 
def count2(n):             
    import math as math
    val = math.log(n,10)   # Log of the base 10 
    return math.ceil(val+1)
```
<h4 style="color:orange;">Find the factorial of a number</h4>

```python
# factorial of a number is given by n*(n-1)*(n-2)*(n-3).......*1 eg. INPUT - 5 OUTPUT - 120 (5*4*3*3*1)
# This is the itreative approch and time complexity is O(n) O(1) auxiliary space
def factorial_iter(n):   
    prod = 1
    for i in range(1,n+1):
        prod = prod*i
    return prod

# This is the recursive approch and time complexity is O(n) and O(n) auxiliary space
def factorial_recursive(n):   
    if(n==0):
        return 1
    return n*factorial_recursive(n-1)

```

<h4 style="color:orange;">Number of trailing zeros in a factorial</h4>

```python
# Trailing zeros in a factorial eg. INPUT - 5 OUTPUT - 1 (120 have 1 zero) or INPUT - 10(3628800 have 2 zero) OUTPUT - 2
# The trailing zeros of a factorial is directly associated with the numbare of 5 in its factor
# 120 - > 1*2*3*4*5 that lead to one zero(2*5)
# our apporoch will be to count the number of 5 in factorial ie. [n/5] + [n/25] + [n/125] ... 
# Time complexity for this solution will be O(log(n))

def trailing_zeros(n):
    val = 0
    count = 5
    while(int(n/count) > 0):
        val = val + int(n/count)
        count = count * 5
    return val
```

<h4 style="color:orange;">Euclidean Algorithm to find GCD(Greatest Common Divisor):</h4>
> Let `a` and `b` are two number and `b` is smaller than `a`
> then the GCD(a,b) = GCD(a-b,b)

```python
# find the GCD eg. INPUT - (10,15) OUTPUT - 5 or INPUT - (50,100) - OUTPUT - 50
# time complexity of this solution is O(log(min(a,b)))
def GCD(a,b):
    if b==0:
        return a
    return GCD(b,a%b)
```
<h4 style="color:orange;">LCM of two number</h4>
> Lcm of two number can be written as `LCM(a,b) = (a*b)/GCD(a,b)`

```python
# find the LCM eg. INPUT - (4,6) OUTPUT - 12 or INPUT - (2,100) OUTPUT - 100
# time complexity of this solution is O(log(min(a,b)))

def LCM(a,b):
    gcd = GCD(a,b)
    return int((a*b)/gcd)
```

<h4 style="color:orange;">Check for prime number</h4>
`The number having only 1 and itself as factor is known as prime number eg. 2,3,5,7,11,37 etc`

```python
# Prime number check eg. INPUT - 121 OUTPUT - False or INPUT - 7 OUTPUT - True
# time complexity for this solution is O(sqrt(n))
 def isPrime(n):
    if n==1:
        return False
    if n==2 or n==3:
        return True
    if n%2==0 or n%3==0:
        return False
    for i in range(5,int(n**(0.5))+1):  
        if(n%i==0 or n%(i+1)==0):
            return False
        i=i+6
    return True
```
<h4 style="color:orange;">Find the prime factors of a number</h4>

```python
# Find the number of prime factors of a number eg. INPUT - 450 OUTPUT - 2*2*3*3*5(450)
# time complexity of this solution is O(sqrt(n))
def primeFactors(n):
    if n<=1:
        return 'no prime factors avilable'
    while(n%2==0):
        print(2,end=" x ")
        n=n//2
    while(n%3==0):
        print(3,end=" x ")
        n=n//3
    for i in range(5,int(n**(0.5))+1):
        while(n%i==0):
            print(i,end=" x ")
            n=n//i
        while(n%(i+1)==0):
            print(i+1,end=" x ")
            n=n//(i+1)
        i=i+6
    if(n>3):
        print(n)
```
<h4 style="color:orange;">Print the divisors of a number in sorted order</h4>

```python
# print the number of divisors in sorted order
# the time complexity for this solution O(sqrt(n))
def divisors(n):
    for i in range(1,int(n**(0.5))):
        if(n%i==0):
            print(i,end=" ")
    for i in range(int(n**(0.5)),0,-1):
        if(n%i==0):
            print(n//i,end=" ")
```
<h4 style="color:orange;">Find list of prime number less than or equal to the given number</h4>

```python
# Sieve of Eratosthenes Algorithm
# find the prime number less than or upto given number INPUT - 10 OUTPUT - 2,3,5,7
# time complexity for this solution is O(nloglog(n))
def prime_number(n):
    data=[True]*(n+1)
    for i in range(2,n+1):
        if(isPrime(i)):
            print(i,end=" ")
            for j in range(i*i,n+1):
                data[j]=False
                j=j+i
     
```