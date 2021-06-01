---
layout: post
title: "MNIST trained model in the FrontEnd"
date: 2021-05-26 05:00:00 -0300
img: mnist.png
description: Taking your Pytorch trained model to the front end with onnx js
tags: [deep-learning, model-in-production]
custom_css: mnist
---

Using ONNX and ONNX.js we can port our *conv nets* to the browser.

This implementation was developed by [Elliot Waite](https://github.com/elliotwaite/pytorch-to-javascript-with-onnx-js). I will try and add the visualization of the feature maps to it in the future.

<div class="card elevation">
    <canvas
    class="canvas elevation"
    id="canvas"
    width="280"
    height="280"></canvas>

<div class="button" id="clear-button">CLEAR</div>

<div class="predictions">
<div class="prediction-col" id="prediction-0">
    <div class="prediction-bar-container">
    <div class="prediction-bar"></div>
    </div>
    <div class="prediction-number">0</div>
</div>

<div class="prediction-col" id="prediction-1">
    <div class="prediction-bar-container">
    <div class="prediction-bar"></div>
    </div>
    <div class="prediction-number">1</div>
</div>

<div class="prediction-col" id="prediction-2">
    <div class="prediction-bar-container">
    <div class="prediction-bar"></div>
    </div>
    <div class="prediction-number">2</div>
</div>

<div class="prediction-col" id="prediction-3">
    <div class="prediction-bar-container">
    <div class="prediction-bar"></div>
    </div>
    <div class="prediction-number">3</div>
</div>

<div class="prediction-col" id="prediction-4">
    <div class="prediction-bar-container">
    <div class="prediction-bar"></div>
    </div>
    <div class="prediction-number">4</div>
</div>

<div class="prediction-col" id="prediction-5">
    <div class="prediction-bar-container">
    <div class="prediction-bar"></div>
    </div>
    <div class="prediction-number">5</div>
</div>

<div class="prediction-col" id="prediction-6">
    <div class="prediction-bar-container">
    <div class="prediction-bar"></div>
    </div>
    <div class="prediction-number">6</div>
</div>

<div class="prediction-col" id="prediction-7">
    <div class="prediction-bar-container">
    <div class="prediction-bar"></div>
    </div>
    <div class="prediction-number">7</div>
</div>

<div class="prediction-col" id="prediction-8">
    <div class="prediction-bar-container">
    <div class="prediction-bar"></div>
    </div>
    <div class="prediction-number">8</div>
</div>

<div class="prediction-col" id="prediction-9">
    <div class="prediction-bar-container">
    <div class="prediction-bar"></div>
    </div>
    <div class="prediction-number">9</div>
</div>
</div>
</div>

<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/onnxjs/dist/onnx.min.js" defer></script>
<script type="text/javascript" src="/assets/js/mnist.js" defer></script>