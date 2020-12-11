import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { SERVER_URL } from '../constants'
import { Country } from '../beans/country'
import 'rxjs/add/operator/catch';

@Injectable({
  providedIn: 'root'
})
export class PredictionService {

  constructor(private http:HttpClient) { }

  httpOptions = {
    headers: new HttpHeaders({
      'Content-Type':  'application/json',
    }),
    responseType: 'blob' as 'json'

  };

  getPredictionGraph(selectedCountry:Country):Observable<Blob> {
        console.log(SERVER_URL);
        return this.http.post<Blob> (SERVER_URL+'/predictor',{
            'country':selectedCountry.name,
            'International_Aviation':selectedCountry.international_Aviation,
            'International_Navigation':selectedCountry.international_Navigation,
            'Chemical_Industry': selectedCountry.chemical_Industry,
            'Domestic_Aviation': selectedCountry.domestic_Aviation,
            'Cropland': selectedCountry.cropland,
            'Forestland': selectedCountry.forestland,
            'Grassland': selectedCountry.grassland,
            'Harvested_Wood_Products': selectedCountry.harvested_Wood_Products,
            'Manufacturing_Industries': selectedCountry.manufacturing_Industries,
            'Metal_Industry': selectedCountry.metal_Industry,
            'Mineral_Industry': selectedCountry.mineral_Industry,
            'Petroleum_Refining': selectedCountry.petroleum_Refining,
            'Public_Electricity_and_Heat_Production': selectedCountry.public_Electricity_and_Heat_Production,
            'Railways': selectedCountry.railways,
            'Road_Transportation': selectedCountry.road_Transportation
          },
          this.httpOptions
        );
    }
}
