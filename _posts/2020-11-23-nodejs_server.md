---
layout: single
header:
  overlay_color: "#333"
  teaser: /assets/images/algorithm/server.png
title:  "Create Your First Server Using Node.js & Express"
excerpt: "Web Development"
breadcrumbs: true
share: true
permalink: /server/
date:    2020-11-23
toc: true

---
Let's create our first `Node.js` server using `express framework`

**Prerequisite**
- `Git Bash or Terminal`
- [Node.js](https://nodejs.org/) `should be installed`
- `Your favourite text editor mine is` [VsCode](https://code.visualstudio.com)


**Let's get our hand dirty with npm**
- Open CMD or Terminal in your system and make first make a workspace ```$ mkdir first_server ```
- Now navigate to our workspace ```$ cd first_server```
- Now its time to [initialize npm](https://docs.npmjs.com/cli/v6/commands/npm-init) in our project to keep track of all dependency via `package.json` file fire command ```$ npm init```
- Provide the required data by `npm init` and this command will create a `package.json` file in working directory
- `package.json` will look like :
![](https://miro.medium.com/max/3336/1*_GLDrXzb_tn02vzBketRQQ.png)


**Let's get our hand dirty with Express**
### Express 
`Fast, unopinionated, minimalist web framework for Node.js` - [Express](http://expressjs.com/)
- Adding Express to your project is only an NPM install away:
```npm
$ npm install express --save
```
- One of the most powerful concepts that Express implements is the middleware pattern.
### Middlewares
You can think of middlewares as Unix pipelines, but for HTTP requests.

![](https://blog-assets.risingstack.com/2016/Apr/express_middlewares_for_building_a_nodejs_http_server-1461068080929.png)

In the diagram we can see how a request can go through an Express application. It travels to three middlewares. Each can modify it, then based on the business logic either the third middleware can send back a response or it can be a route handler.
- **Let's write code for server using express middlewares**

```javascript
const express = require("express");
const app = express();
const PORT = process.env.PORT || 8000;

app.get("/",(req,res)=>{
return res.send("<h1>Hello World from our first Server</h1>")
});

app.listen(PORT,(err)=>{
if(err){
    console.log('Error in creating server',err);
}
else{
    console.log('Server is running at port: ',PORT);
}
});
```
> Now open terminal and fire command `node index.js` and open browser and search for `http://localhost:8000` or `http://127.0.0.1:8000/`
> Bam in browser you will see `Hello World from our first Server`

#### Happy Coding