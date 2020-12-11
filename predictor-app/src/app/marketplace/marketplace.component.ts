import { Component, OnInit } from '@angular/core';
import { MarketplaceService } from '../services/marketplace.service'
import { Observable } from 'rxjs';
import { FormBuilder } from "@angular/forms";
import { MatButtonModule } from '@angular/material/button';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

@Component({
  selector: 'app-marketplace',
  templateUrl: './marketplace.component.html',
  styleUrls: ['./marketplace.component.scss']
})
export class MarketplaceComponent implements OnInit {

  constructor(private marketplaceService: MarketplaceService,public fb: FormBuilder) { }

  downloadModelForm = this.fb.group({
    feature: ['Railways']
  })

  downloadDataForm = this.fb.group({
    data: ['DataMainImputedModified']
  })

  get myForm() {
    return this.downloadModelForm.get('feature');
    return this.downloadDataForm.get('data');
  }

  ngOnInit(): void {
  }

  onClickDownload():void{
    this.marketplaceService.getRequestedFile(this.downloadModelForm.get('feature').value).subscribe( data => {
      console.log(data);
      this.createZipFromBlob(data);

    },
    error => console.log('Error from backend API', +error));
  }

  onClickDownloadData():void{
    this.marketplaceService.getRequestedDataFile(this.downloadDataForm.get('data').value).subscribe( data => {
      console.log(data);
      this.createXLSXFromBlob(data);

    },
    error => console.log('Error from backend API', +error));
  }

  createZipFromBlob(data: Blob) {
     const blob = new Blob([data], {
        type: 'application/zip'
     });
     const url = window.URL.createObjectURL(blob);
     window.open(url);
  }

  createXLSXFromBlob(data: Blob) {
     const blob = new Blob([data], {
        type: 'application/vnd.ms-excel'
     });
     const url = window.URL.createObjectURL(blob);
     window.open(url);
  }
}
