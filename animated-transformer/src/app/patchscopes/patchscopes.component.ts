import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { Router, ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-patchscopes',
  templateUrl: './patchscopes.component.html',
  styleUrl: './patchscopes.component.scss'
})
export class PatchscopesComponent implements OnInit {
  path_to_model = '';

  constructor(private route: ActivatedRoute, private router: Router) {
    console.log(`tf.getBackend: ${tf.getBackend()}`);
  }

  ngOnInit() {
    this.route.queryParams.subscribe((params) => {
      this.path_to_model = params['path_to_model'] || '';
    });
  }

  updatePathToModel(input: string) {
    const queryParams = { input };
    this.router.navigate([], {
      relativeTo: this.route,
      queryParams: queryParams,
      // remove to replace all query params by provided
      queryParamsHandling: 'merge',
    });
  }
}
