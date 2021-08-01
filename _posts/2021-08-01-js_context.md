---
layout: single
header:
  overlay_color: "#333"
  teaser: https://static.javatpoint.com/images/javascript/javascript_logo.png
title:  "How Javascript Execuion Context Work?"
excerpt: "Web Development"
breadcrumbs: true
share: true
permalink: /execution-context/
date:    2021-08-01
toc: true

---
## Everything in js happen inside execution context

![](https://miro.medium.com/max/2000/1*ACtBy8CIepVTOSYcVwZ34Q.png)

## Execution context:
- Memory -> Variable Enivornment
    - Variables and Functions are stored as key value pair
- Code -> Thread of Execution
    - JS is synchronous single-threaded Language
    - One line code execution at a time



## What Happens when you run a js program ?

```js
var n = 2;
function square(num){
    var ans = num*num;
    return ans;
}
var square2 = square(n);
var square4 = square(4);
```
Ans:
- A Global Context is Created

| Memory (Phase-1)      |    Code Phase(2)          |
| :-------------        | :----------:              |
| n : undefined         | n : 2 -> Memory           | 
| square : {Whole code} | Nothing to Do             |
| square2 : undefined   | New Execution Context is created (Memory & Code) |
|square4 : undefined    | New Execution Context is Created (Memory & Code) |

- New Execution context (square2)

| Memory (Phase-1)      |    Code Phase(2)    |
| :-------------        | :----------:        |
| num : undefined       | num : 2 -> Memory   |     
| ans : undefined       | ans : 4  -> return 4 to global context in memory  at square2|

- Changes Reflected in Global Context (square2)

| Memory (Phase-1)      |    Code Phase(2)          |
| :-------------        | :----------:              |
| n : 2                 | n : 2 -> Memory           | 
| square : {Whole code} | Nothing to Do             |
| square2 : 4           | Data replaced from square2 context |
|square4 : undefined    | New Execution Context is Created (Memory & Code) |

- New Execution context (square4)

| Memory (Phase-1)      |    Code Phase(2)    |
| :-------------        | :----------:        |
| num : undefined       | num : 4 -> Memory   |     
| ans : undefined       | ans : 16  -> return 16 to global context in memory at square4|

- Changes Reflected in Global Context (square4)

| Memory (Phase-1)      |    Code Phase(2)          |
| :-------------        | :----------:              |
| n : 4                 | n : 2 -> Memory           | 
| square : {Whole code} | Nothing to Do             |
| square2 : 4           | Data replaced from square2 context |
|square4 : 16           | Data replaced from square4 context |


## Js manages all the execution context with help of js call stack

- When we run a js program then whole global context is pushed into the stack.
- Then function execution start and local context is pushed to stack.
- When the work of execution context is completed the it is poped out of the stack.
