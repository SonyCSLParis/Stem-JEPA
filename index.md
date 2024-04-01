<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


<script type="text/javascript"> 
      // Show button
      function look(type){ 
      param=document.getElementById(type); 
      if(param.style.display == "none") param.style.display = "block"; 
      else param.style.display = "none" 
      } 
</script> 

# Stem-JEPA - Supplementary material

This repository describes the additional material and experiments around the paper "Joint-Embedding Predictive Architecture for Musical Stem Affinity Estimation", submitted to ISMIR 2024.

put abstract here

put figure 1 here



## Additional results on downstream tasks

- confusion matrices

## Audio examples

TODO

<table>
<caption><b> Tom </b></caption>
  <tr>
    <td style="text-align: center; vertical-align: middle;"><b>Input</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>WAE Output</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>Output</b></td>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls>
      <source src="https://anonymous9123.github.io/iccc-ndm/sounds/rec/tom.wav">
      </audio>
    </td>
    <td> </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls>
      <source src="https://anonymous9123.github.io/iccc-ndm/sounds/rec/tomr.wav">
      </audio>
    </td>
  </tr>

  <tr>
    <td><img class="recimg" src="https://anonymous9123.github.io/iccc-ndm/figures/rec/tomin.png"></td>
    <td><img class="recimg" src="https://anonymous9123.github.io/iccc-ndm/figures/rec/tomWAE.png"></td>
    <td><img class="recimg" src="https://anonymous9123.github.io/iccc-ndm/figures/rec/tomout.png"></td>
  </tr>
  <tr>
    <td><img class="recimg" src="https://anonymous9123.github.io/iccc-ndm/figures/rec/tomwin.png"></td>
    <td></td>
    <td><img class="recimg" src="https://anonymous9123.github.io/iccc-ndm/figures/rec/tomwout.png"></td>
  </tr>
</table>

<table>
<caption><b> Conga </b></caption>
  <tr>
    <td style="text-align: center; vertical-align: middle;"><b>Input</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>WAE Output</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>Output</b></td>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls>
      <source src="https://anonymous9123.github.io/iccc-ndm/sounds/rec/conga.wav">
      </audio>
    </td>
    <td> </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls>
      <source src="https://anonymous9123.github.io/iccc-ndm/sounds/rec/congar.wav">
      </audio>
    </td>
  </tr>

  <tr>
    <td><img class="recimg" src="https://anonymous9123.github.io/iccc-ndm/figures/rec/cgin.png"></td>
    <td><img class="recimg" src="https://anonymous9123.github.io/iccc-ndm/figures/rec/cgWAE.png"></td>
    <td><img class="recimg" src="https://anonymous9123.github.io/iccc-ndm/figures/rec/cgout.png"></td>
  </tr>
  <tr>
    <td><img class="recimg" src="https://anonymous9123.github.io/iccc-ndm/figures/rec/cgwin.png"></td>
    <td></td>
    <td><img class="recimg" src="https://anonymous9123.github.io/iccc-ndm/figures/rec/cgwout.png"></td>
  </tr>
</table>

<table>
<caption><b> Clap </b></caption>
  <tr>
    <td style="text-align: center; vertical-align: middle;"><b>Input</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>WAE Output</b></td>
    <td style="text-align: center; vertical-align: middle;"><b>Output</b></td>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls>
      <source src="https://anonymous9123.github.io/iccc-ndm/sounds/rec/clap.wav">
      </audio>
    </td>
    <td> </td>
    <td style="text-align: center; vertical-align: middle;">
      <audio controls>
      <source src="https://anonymous9123.github.io/iccc-ndm/sounds/rec/clapr.wav">
      </audio>
    </td>
  </tr>

  <tr>
    <td><img class="recimg" src="https://anonymous9123.github.io/iccc-ndm/figures/rec/clapin.png"></td>
    <td><img class="recimg" src="https://anonymous9123.github.io/iccc-ndm/figures/rec/clapWAE.png"></td>
    <td><img class="recimg" src="https://anonymous9123.github.io/iccc-ndm/figures/rec/clapout.png"></td>
  </tr>
  <tr>
    <td><img class="recimg" src="https://anonymous9123.github.io/iccc-ndm/figures/rec/clapwin.png"></td>
    <td></td>
    <td><img class="recimg" src="https://anonymous9123.github.io/iccc-ndm/figures/rec/clapwout.png"></td>
  </tr>
</table>
