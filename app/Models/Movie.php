<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Movie extends Model
{
    public $fillable = ['overview', 'title', 'results'];
}
