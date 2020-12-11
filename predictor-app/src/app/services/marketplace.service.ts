import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { SERVER_URL } from '../constants'

@Injectable({
  providedIn: 'root'
})
export class MarketplaceService {

  constructor(private http:HttpClient) { }

  httpOptions = {
    headers: new HttpHeaders({
      'Content-Type':  'application/json',
    }),
    responseType: 'blob' as 'json'

  };
  getRequestedFile(feature: string):Observable<Blob>{
    console.log(SERVER_URL);
    return this.http.post<Blob> (
      SERVER_URL+'/marketplace_models',
      {'feature':feature},
      this.httpOptions
    );
  }

  getRequestedDataFile(data: string):Observable<Blob>{
    console.log(SERVER_URL);
    return this.http.post<Blob> (
      SERVER_URL+'/marketplace_data',
      {'data':data},
      this.httpOptions
    );
  }
}
