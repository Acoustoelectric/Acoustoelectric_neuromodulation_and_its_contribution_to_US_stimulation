# Acoustoelectric neuromodulation and its contribution to US stimulation

## Overview

The code contained in this repository, works in conjunction with the datasets uploaded to figshare at: DOI: https://doi.org/10.6084/m9.figshare.c.7909283

The purpose of this code and data is to recreate the figures in the article 'Non-invasive in vivo acoustoelectric neuromodulation and its contribution to ultrasound stimulation'

## Abstract

Non-invasive brain stimulation offers therapeutic potential without the risks of surgery, yet current electrical approaches lack spatial precision and depth due to the long wavelengths of electric fields. Here we demonstrate that the acoustoelectric interaction—the nonlinear coupling between applied acoustic and electric fields—can overcome these limitations to achieve spatially focused, non-invasive neuromodulation. Using in vitro and in vivo rodent electrophysiology, we show motor-evoked potentials that depend on both the amplitude and frequency of the acoustoelectric field, with artefactual controls excluding purely acoustic or electrical origins. We further identify an acoustoelectric contribution to conventional ultrasound stimulation, arising from the interaction between ultrasound-induced electrical signals and propagating acoustic waves. These findings establish acoustoelectric neuromodulation as a distinct mechanism of neural activation and a significant contributor to how ultrasound stimulation influences brain activity, opening new directions for precise, non-invasive neuromodulation and neurotherapeutic development.

## System Requirements

+  macOS: Sequoia 15.6.1

```
Python 3.13.7

```

## Installation Guide

```
pip install -r requirements.txt

```
If you have any problems, it's likely that the requirements installation didn't complete, and you have incorrect versions of the modules installed. The following command will check the version of a single module, and you should check each module to ensure it is the same as listed in the requirements.txt file. 

```
pip show <package_name>

```

Alternatively you can run the following command to get a list of all installed modules. Before creating an issue in this repository regarding the installation process, check each version in the requirements.txt against your installed version. They should match! 

```
pip freeze

```

## License

This project is covered under the <b>Apache 2.0 License</b>.