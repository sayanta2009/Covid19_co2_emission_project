import { Component, OnInit, ViewEncapsulation } from '@angular/core';
import { FormGroup, FormControl } from '@angular/forms';
import { Country } from '../beans/country'
import { PredictionService } from '../services/prediction.service'
import { Observable } from 'rxjs';
import { DomSanitizer } from '@angular/platform-browser';
import { COUNTRIES_LIST, FEATURES_LIST } from '../constants'
import { MatSelectModule } from '@angular/material/select';
import { MatButtonModule } from '@angular/material/button';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatCardModule } from '@angular/material/card';



@Component({
  selector: 'app-predictions',
  templateUrl: './predictions.component.html',
  styleUrls: ['./predictions.component.scss'],
  encapsulation: ViewEncapsulation.None
})
export class PredictionsComponent implements OnInit {

  graphState:boolean;
  predicting:boolean;
  selectedCountry = new Country();
  country_name = new FormControl('');
  international_Aviation = new FormControl('');
  international_Navigation = new FormControl('');
  chemical_Industry = new FormControl('');
  domestic_Aviation = new FormControl('');
  manufacturing_Industries = new FormControl('');
  metal_Industry = new FormControl('');
  mineral_Industry = new FormControl('');
  petroleum_Refining = new FormControl('');
  public_Electricity_and_Heat_Production = new FormControl('');
  railways = new FormControl('');
  road_Transportation = new FormControl('');
  countries = COUNTRIES_LIST;
  features = FEATURES_LIST;
  imageToShow: any;
  constructor(private predictionService: PredictionService,private sanitizer: DomSanitizer) {
  }

  ngOnInit(): void {
    this.country_name.setValue('Germany');
    this.international_Aviation.setValue(0);
    this.international_Navigation.setValue(0);
    this.chemical_Industry.setValue(0);
    this.domestic_Aviation.setValue(0);
    this.manufacturing_Industries.setValue(0);
    this.metal_Industry.setValue(0);
    this.mineral_Industry.setValue(0);
    this.petroleum_Refining.setValue(0);
    this.public_Electricity_and_Heat_Production.setValue(0);
    this.railways.setValue(0);
    this.road_Transportation.setValue(0);
    this.graphState = false;
    this.predicting = false;
  }

  onClickPredict(): void{
    this.selectedCountry.name = this.country_name.value;
    this.selectedCountry.international_Aviation = this.international_Aviation.value;
    this.selectedCountry.international_Navigation = this.international_Navigation.value;
    this.selectedCountry.chemical_Industry = this.chemical_Industry.value;
    this.selectedCountry.domestic_Aviation = this.domestic_Aviation.value;
    this.selectedCountry.manufacturing_Industries = this.manufacturing_Industries.value;
    this.selectedCountry.metal_Industry = this.metal_Industry.value;
    this.selectedCountry.mineral_Industry = this.mineral_Industry.value;
    this.selectedCountry.petroleum_Refining = this.petroleum_Refining.value;
    this.selectedCountry.public_Electricity_and_Heat_Production = this.public_Electricity_and_Heat_Production.value;
    this.selectedCountry.railways = this.railways.value;
    this.selectedCountry.road_Transportation = this.road_Transportation.value;
    this.selectedCountry.cropland = 0;
    this.selectedCountry.forestland = 0;
    this.selectedCountry.grassland = 0;
    this.selectedCountry.harvested_Wood_Products = 0;
    console.log(this.selectedCountry);
    this.getPredictionGraph();
    this.graphState = false;
    this.predicting = true;
  }

  getPredictionGraph(): void {
    this.predictionService.getPredictionGraph(this.selectedCountry).subscribe( data => {
      console.log(data);
      this.createImageFromBlob(data);
      this.graphState = true;
      this.predicting = false;
    },
    error => {
      console.log('Error from backend API', +error);
      this.graphState = false;
      this.predicting = false;
    });

  }

  createImageFromBlob(image: Blob) {
     let reader = new FileReader();
     reader.addEventListener("load", () => {
        this.imageToShow = reader.result;
     }, false);

     if (image) {
        reader.readAsDataURL(image);
     }
  }

}
