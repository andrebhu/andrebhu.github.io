---
layout: post
title: MapleCTF 2022 Web Writeups
date: 2022-04-19
---
- [honksay [140 solves]](#honksay-140-solves)
- [Pickle Factory [66 solves]](#pickle-factory-66-solves)
- [Bookstore [60 solves]](#bookstore-60-solves)

### honksay [140 solves]
*Haha goose say funny thing*

By looking through the source, the application is running Express, a Node.js framework, and has the following routes:

- `/`: has some interesting cookie logic
- `/changehonk`: sets the `honk` cookie to the `newhonk` query and `honkcount` to 0
- `/report`: calls the goose, uses [Puppeteer](https://pptr.dev/)

Let's take a deeper look at `/`:
```javascript
app.get('/', (req, res) => {
    if (req.cookies.honk){
        //construct object
        let finalhonk = {};
        if (typeof(req.cookies.honk) === 'object'){
            finalhonk = req.cookies.honk
        } else {
            finalhonk = {
                message: clean(req.cookies.honk), 
                amountoftimeshonked: req.cookies.honkcount.toString()
            };
        }
        res.send(template(finalhonk.message, finalhonk.amountoftimeshonked));
    } else {
        const initialhonk = 'HONK';
        res.cookie('honk', initialhonk, {
            httpOnly: true
        });
        res.cookie('honkcount', 0, {
            httpOnly: true
        });
        res.redirect('/');
    }
});
```





```javascript
app.get('/changehonk', (req, res) => {
    res.cookie('honk', req.query.newhonk, {
        httpOnly: true
    });
    res.cookie('honkcount', 0, {
        httpOnly: true
    });
    res.redirect('/');
});
```



### Pickle Factory [66 solves]
*My cousin said he once got fired for putting his p\*ckle into the pickle slicer at his old workplace. Can you confirm that it's true for me?*

### Bookstore [60 solves]
*Maple Stores 2 is out! Get it for me pwease*